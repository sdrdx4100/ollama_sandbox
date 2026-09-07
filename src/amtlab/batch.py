"""複数ログファイルの一括読み込みと解析。

parquet / CSV が何十〜何百ファイルあっても、

    1. スキーマ(列名)だけ先に読む
    2. 必要な信号に対応する列だけを読み込む(parquet の列指向を活かす)
    3. ファイルごとに変速イベントを切り出して KPI 行に畳む
    4. 時系列は捨て、KPI 表だけを結合する

という流れでメモリを使わずに回す。ファイル間で信号構成が違う場合は
「どのファイルにどの信号があるか」の表を出して差分を確認できる。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from . import j1939
from .ingest import SignalMap, detect_shift_events, event_kpis, standardize
from .simulation.vehicle import VehicleParams

#: 対応するログ拡張子
LOG_SUFFIXES = (".parquet", ".pq", ".csv", ".csv.gz", ".feather", ".arrow")


def iter_log_paths(patterns: str | Path | list[str | Path]) -> list[Path]:
    """ファイル / ディレクトリ / glob パターンを展開してパス一覧にする。

    ディレクトリを渡した場合は再帰的に対応拡張子のファイルを集める。
    """

    if isinstance(patterns, (str, Path)):
        patterns = [patterns]

    found: list[Path] = []
    for pattern in patterns:
        path = Path(pattern)
        if path.is_dir():
            found += [
                p for p in sorted(path.rglob("*"))
                if p.is_file() and _has_log_suffix(p)
            ]
        elif path.exists():
            found.append(path)
        else:  # glob パターン
            base = path.parent if str(path.parent) not in ("", ".") else Path(".")
            found += sorted(base.glob(path.name))
    unique = sorted({p.resolve() for p in found if p.is_file()})
    return unique


def _has_log_suffix(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(suffix) for suffix in LOG_SUFFIXES)


def _require_pyarrow(module: str):
    """parquet / feather を読むための pyarrow を読み込む。"""
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - 環境依存
        raise ImportError(
            "parquet / feather の読み込みには pyarrow が必要です: "
            'pip install "amtlab[parquet]"'
        ) from exc


def read_log_schema(path: str | Path) -> list[str]:
    """データ本体を読まずに列名だけ取得する。"""

    path = Path(path)
    name = path.name.lower()
    if name.endswith((".parquet", ".pq")):
        pq = _require_pyarrow("pyarrow.parquet")
        return [str(c) for c in pq.ParquetFile(path).schema_arrow.names]
    if name.endswith((".feather", ".arrow")):
        feather = _require_pyarrow("pyarrow.feather")
        return [str(c) for c in feather.read_table(path, memory_map=True).schema.names]
    return [str(c) for c in pd.read_csv(path, nrows=0).columns]


def read_log(path: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    """ログを読み込む(列を指定すればその列だけ)。"""

    path = Path(path)
    name = path.name.lower()
    if name.endswith((".parquet", ".pq")):
        return pd.read_parquet(path, columns=columns)
    if name.endswith((".feather", ".arrow")):
        feather = _require_pyarrow("pyarrow.feather")
        return feather.read_feather(path, columns=columns)
    df = pd.read_csv(path)
    return df[columns] if columns else df


def load_log(
    path: str | Path, signal_map: SignalMap | None = None
) -> tuple[pd.DataFrame, dict[str, str]]:
    """必要な信号の列だけを読み込む。

    先にスキーマから対応表を作り、使う列だけを読むので、
    数百列あるログでもメモリと I/O を無駄にしない。
    """

    sm = signal_map or SignalMap()
    schema = read_log_schema(path)
    probe = pd.DataFrame(columns=schema)
    mapping = sm.resolve(probe)  # 列が足りなければここで KeyError
    df = read_log(path, columns=sorted(set(mapping.values())))
    return df, mapping


@dataclass
class FileReport:
    """1 ファイル分の読み込み結果。"""

    path: str
    n_rows: int = 0
    duration_s: float = 0.0
    n_events: int = 0
    detected: tuple[str, ...] = ()
    jerk_quality: str = ""
    detection_source: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error


@dataclass
class BatchResult:
    """複数ファイルの解析結果。"""

    events: pd.DataFrame = field(default_factory=pd.DataFrame)
    files: pd.DataFrame = field(default_factory=pd.DataFrame)
    warnings: list[str] = field(default_factory=list)

    @property
    def n_files(self) -> int:
        return len(self.files)

    @property
    def n_ok(self) -> int:
        return int((self.files["error"] == "").sum()) if len(self.files) else 0

    def failures(self) -> pd.DataFrame:
        if not len(self.files):
            return pd.DataFrame()
        return self.files[self.files["error"] != ""]

    def summary(self) -> str:
        lines = [
            f"{self.n_ok}/{self.n_files} ファイル読み込み成功 / "
            f"変速イベント {len(self.events)} 件",
        ]
        if len(self.events):
            per_file = self.files.loc[self.files["error"] == "", "n_events"]
            lines.append(
                f"1 ファイルあたり {per_file.mean():.1f} 件 "
                f"(最小 {per_file.min()} / 最大 {per_file.max()})"
            )
            qualities = sorted(set(self.events.get("jerk_quality", pd.Series(dtype=str))))
            if qualities:
                lines.append("jerk 信頼度: " + ", ".join(map(str, qualities)))
        for warning in self.warnings:
            lines.append(f"  - {warning}")
        return "\n".join(lines)


def _analyze_one(
    path: Path,
    signal_map: SignalMap | None,
    vehicle: VehicleParams | None,
    resample_hz: float | None,
    settle_time: float,
) -> tuple[FileReport, pd.DataFrame]:
    report = FileReport(path=str(path))
    try:
        df, mapping = load_log(path, signal_map)
        report.n_rows = len(df)
        report.detected = tuple(sorted(mapping))
        trace = standardize(df, signal_map, resample_hz=resample_hz)
        report.duration_s = float(trace["time"].iloc[-1]) if len(trace) else 0.0
        report.jerk_quality = str(trace.attrs.get("jerk_quality", ""))

        events = detect_shift_events(trace, vehicle, settle_time=settle_time)
        rows = [event_kpis(trace, ev, vehicle) for ev in events]
        kpi = pd.DataFrame([r for r in rows if r])
        if len(kpi):
            kpi.insert(0, "source_file", path.name)
            kpi.insert(1, "source_path", str(path))
            report.detection_source = str(kpi["detection_source"].iloc[0])
        report.n_events = len(kpi)
        return report, kpi
    except Exception as exc:  # ファイル 1 つの失敗で全体を止めない
        report.error = f"{type(exc).__name__}: {exc}"
        return report, pd.DataFrame()


def analyze_logs(
    patterns: str | Path | list[str | Path],
    signal_map: SignalMap | None = None,
    vehicle: VehicleParams | None = None,
    resample_hz: float | None = 100.0,
    settle_time: float = 0.6,
    n_jobs: int = -1,
) -> BatchResult:
    """複数ログをまとめて解析し、変速イベント KPI 表に畳む。

    時系列はファイルごとに捨てるので、ファイル数を増やしてもメモリは
    イベント数にしか比例しない。
    """

    paths = iter_log_paths(patterns)
    if not paths:
        return BatchResult(warnings=[f"ログが見つかりません: {patterns}"])

    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_analyze_one)(p, signal_map, vehicle, resample_hz, settle_time)
        for p in paths
    )
    reports = [r for r, _ in results]
    frames = [k for _, k in results if len(k)]

    events = (
        pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    )
    if len(events):
        events.insert(0, "global_event_id", np.arange(len(events)))
    files = pd.DataFrame([r.__dict__ for r in reports])
    files["detected"] = files["detected"].apply(lambda t: ",".join(t))

    result = BatchResult(events=events, files=files)
    result.warnings = _batch_warnings(reports, events)
    return result


def _batch_warnings(reports: list[FileReport], events: pd.DataFrame) -> list[str]:
    warnings: list[str] = []
    failed = [r for r in reports if r.error]
    if failed:
        warnings.append(
            f"読み込みに失敗: {len(failed)} ファイル(例: "
            f"{Path(failed[0].path).name} — {failed[0].error})"
        )
    empty = [r for r in reports if r.ok and r.n_events == 0]
    if empty:
        warnings.append(
            f"変速イベントが 0 件のファイル: {len(empty)} 件(例: "
            f"{Path(empty[0].path).name})"
        )

    signatures = {r.detected for r in reports if r.ok}
    if len(signatures) > 1:
        warnings.append(
            f"信号構成がファイル間で {len(signatures)} 種類あります。"
            "signal_presence() で差分を確認してください"
        )
    sources = {r.detection_source for r in reports if r.ok and r.detection_source}
    if len(sources) > 1:
        warnings.append(
            "変速イベントの切り出し方法がファイル間で混在しています "
            f"({', '.join(sorted(sources))})。変速時間の定義が揃わないため"
            "比較には注意が必要です"
        )
    if len(events) and "jerk_quality" in events.columns:
        if set(events["jerk_quality"]) - {"measured", "derived"}:
            warnings.append(
                "ジャークが参考値のファイルが含まれます"
                "(jerk_quality 列で絞り込んでください)"
            )
    return warnings


def signal_presence(
    patterns: str | Path | list[str | Path], signal_map: SignalMap | None = None
) -> pd.DataFrame:
    """ファイル × 信号の有無を表にする(データ本体は読まない)。

    ファイルごとに取れている信号が違う場合の差分確認に使う。
    """

    sm = signal_map or SignalMap()
    paths = iter_log_paths(patterns)
    keys = [signal.key for signal in j1939.J1939_CATALOG]
    rows = []
    for path in paths:
        row: dict[str, object] = {"file": path.name, "n_columns": 0, "error": ""}
        row.update({key: False for key in keys})
        try:
            schema = read_log_schema(path)
            probe = pd.DataFrame(columns=schema)
            mapping = j1939.detect_columns(probe) if sm.auto_detect else {}
            for f_name in (f.name for f in _signal_fields()):
                explicit = getattr(sm, f_name, None)
                if explicit is not None and explicit in schema:
                    mapping[f_name] = explicit
            row["n_columns"] = len(schema)
            row.update({key: key in mapping for key in keys})
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    frame = pd.DataFrame(rows)
    for key in keys:
        frame[key] = frame[key].astype(bool)
    return frame


def _signal_fields():
    from dataclasses import fields as dc_fields

    return [f for f in dc_fields(SignalMap) if f.name != "auto_detect"]


def inspect_logs(
    patterns: str | Path | list[str | Path], limit: int | None = None
) -> pd.DataFrame:
    """複数ログのチャネル診断(実効サンプルレート)をまとめる。

    レートの推定にはデータ本体が要るため、``limit`` で対象ファイル数を
    絞れるようにしている(既定は全ファイル)。
    """

    paths = iter_log_paths(patterns)
    if limit:
        paths = paths[:limit]
    frames = []
    for path in paths:
        try:
            df, mapping = load_log(path)
            report = j1939.inspect_log(df, mapping)
            frame = report.to_frame()
            frame.insert(0, "file", path.name)
            frames.append(frame)
        except Exception as exc:
            frames.append(
                pd.DataFrame([{"file": path.name, "error": f"{type(exc).__name__}: {exc}"}])
            )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
