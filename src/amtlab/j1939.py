"""J1939 信号の定義・自動検出・物理量への変換。

実車ログ(DBC でデコード済みの CSV / parquet)を想定し、SPN 単位で
列を突き合わせる。列名はロガーやツールによって英語名・日本語名・SPN 番号など
まちまちなので、別名の正規表現でゆるく自動検出する。

対応 SPN(このプロジェクトで使うもの):

======  ======  ==========================================  ===================
SPN     PGN     信号                                        用途
======  ======  ==========================================  ===================
91      EEC2    Accelerator Pedal Position 1                運転条件(負荷)
190     EEC1    Engine Speed                                回転同期の評価
512     EEC1    Driver's Demand Engine - Percent Torque      トルク復帰の目標
513     EEC1    Actual Engine - Percent Torque              トルクダウン/復帰
544     EC1     Engine Reference Torque                     % → Nm 換算の基準
84      CCVS1   Wheel-Based Vehicle Speed                   車速の落ち込み
904     EBC2    Front Axle Speed                            車速(前輪)
161     ETC1    Transmission Input Shaft Speed              クラッチすべり
191     ETC1    Transmission Output Shaft Speed             変速機出力回転
522     ETC1    Percent Clutch Slip                         クラッチ締結の進行
574     ETC1    Transmission Shift In Process               変速イベントの切り出し
523     ETC2    Transmission Current Gear                   カレントギア
524     ETC2    Transmission Selected Gear                  変速先ギヤ
526     ETC2    Transmission Actual Gear Ratio              実ギヤ比(N 判定)
======  ======  ==========================================  ===================

.. note::
   SPN 513 は「参照トルクに対する出力トルクの割合」であり、より正確には
   SPN 514(Nominal Friction - Percent Torque)を差し引いた値が正味の
   フライホイールトルクになる。514 がログにあれば自動で使用する。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class J1939Signal:
    """1 つの SPN と、ログ列名の別名パターン。"""

    key: str  # amtlab 内部の正規化名
    spn: int | None
    pgn: str
    name_en: str
    name_ja: str
    unit: str
    aliases: tuple[str, ...]  # 正規表現(小文字化した列名に対して検索)
    typical_rate_hz: float | None = None
    discrete: bool = False  # 離散信号(ギヤ位置・状態フラグ)は更新周期を推定しない

    def matches(self, column: str) -> bool:
        norm = _normalize(column)
        if self.spn is not None and re.search(rf"(^|[^0-9]){self.spn}([^0-9]|$)", norm):
            return True
        return any(re.search(pattern, norm) for pattern in self.aliases)


def _normalize(name: str) -> str:
    """列名を突き合わせ用に正規化する(小文字化・記号を _ に統一)。"""
    text = str(name).strip().lower()
    text = re.sub(r"[\s\-\./\[\]\(\)（）:：]+", "_", text)
    return re.sub(r"_+", "_", text)


#: 対応 SPN のカタログ
J1939_CATALOG: tuple[J1939Signal, ...] = (
    J1939Signal(
        "time", None, "-", "Timestamp", "時刻", "s",
        (r"^time", r"timestamp", r"^t$", r"時刻", r"経過時間"),
    ),
    J1939Signal(
        "accel_pedal_pct", 91, "EEC2", "Accelerator Pedal Position 1",
        "アクセルペダルポジション", "%",
        # SPN 51(Engine Throttle Position)は別信号だが、実用上ここに寄せる
        (r"accel(erator)?_?pedal", r"app\d?$", r"^throttle", r"アクセル", r"ペダル開度"),
        20.0,
    ),
    J1939Signal(
        "demand_torque_pct", 512, "EEC1", "Driver's Demand Engine - Percent Torque",
        "ドライバー要求トルク", "%",
        (r"driver.*demand.*torque", r"demand.*torque", r"要求トルク", r"ドライバ"), 50.0,
    ),
    J1939Signal(
        "actual_torque_pct", 513, "EEC1", "Actual Engine - Percent Torque",
        "実際のエンジントルク率", "%",
        (r"actual.*engine.*torque", r"actual.*torque", r"実.*トルク率",
         r"エンジントルク率"), 50.0,
    ),
    J1939Signal(
        "friction_torque_pct", 514, "EEC1", "Nominal Friction - Percent Torque",
        "フリクショントルク率", "%",
        (r"nominal.*friction", r"friction.*torque", r"フリクション"), 50.0,
    ),
    J1939Signal(
        "reference_torque_nm", 544, "EC1", "Engine Reference Torque",
        "エンジン参照トルク", "Nm",
        (r"reference.*torque", r"ref_?torque", r"参照トルク"), 0.2,
    ),
    J1939Signal(
        "engine_speed_rpm", 190, "EEC1", "Engine Speed", "エンジン回転数", "rpm",
        (r"engine_?speed", r"^ne$", r"^rpm$", r"エンジン回転"), 50.0,
    ),
    J1939Signal(
        "speed_kmh", 84, "CCVS1", "Wheel-Based Vehicle Speed",
        "ホイールベース車速", "km/h",
        (r"wheel_?based.*speed", r"vehicle_?speed", r"^speed_?km", r"^speed$",
         r"車速", r"^vsp"), 10.0,
    ),
    J1939Signal(
        "front_axle_speed_kmh", 904, "EBC2", "Front Axle Speed", "前輪速度", "km/h",
        (r"front_?axle_?speed", r"front_?wheel", r"前輪"), 20.0,
    ),
    J1939Signal(
        "input_shaft_rpm", 161, "ETC1", "Transmission Input Shaft Speed",
        "インプットシャフト回転数", "rpm",
        (r"input_?shaft", r"インプットシャフト", r"入力軸"), 50.0,
    ),
    J1939Signal(
        "output_shaft_rpm", 191, "ETC1", "Transmission Output Shaft Speed",
        "アウトプットシャフト回転数", "rpm",
        (r"output_?shaft", r"アウトプットシャフト", r"出力軸"), 50.0,
    ),
    J1939Signal(
        "clutch_slip_pct", 522, "ETC1", "Percent Clutch Slip", "クラッチ滑り率", "%",
        (r"clutch_?slip", r"percent_?clutch", r"クラッチ.*滑り", r"クラッチ.*スリップ"), 50.0,
    ),
    J1939Signal(
        "shift_in_process", 574, "ETC1", "Transmission Shift In Process",
        "トランスシフトインプロセス", "-",
        (r"shift_?in_?process", r"shift_?in_?prog", r"シフトインプロセス",
         r"変速中"), 50.0, discrete=True,
    ),
    J1939Signal(
        "gear", 523, "ETC2", "Transmission Current Gear", "現在のギア位置", "gear",
        (r"current_?gear", r"actual_?gear(?!_?ratio)", r"^gear$", r"^gear_?pos",
         r"現在.*ギア", r"カレントギア"), 20.0, discrete=True,
    ),
    J1939Signal(
        "selected_gear", 524, "ETC2", "Transmission Selected Gear",
        "選択されたギア位置", "gear",
        (r"selected_?gear", r"target_?gear", r"選択.*ギア"), 20.0, discrete=True,
    ),
    J1939Signal(
        "gear_ratio", 526, "ETC2", "Transmission Actual Gear Ratio",
        "実際のギア比", "-",
        (r"gear_?ratio", r"ギア比", r"ギヤ比"), 20.0,
    ),
)

CATALOG_BY_KEY: dict[str, J1939Signal] = {s.key: s for s in J1939_CATALOG}

#: 変速イベントの切り出しに最低限必要な信号(いずれかの組み合わせ)
MINIMUM_KEYS = ("time", "engine_speed_rpm", "speed_kmh")


def detect_columns(df: pd.DataFrame) -> dict[str, str]:
    """DataFrame の列名から SPN を自動検出して {内部名: 列名} を返す。

    複数列が同じ SPN に該当した場合は、より短い(修飾の少ない)列名を採用する。
    """

    found: dict[str, str] = {}
    for signal in J1939_CATALOG:
        candidates = [c for c in df.columns if signal.matches(c)]
        if candidates:
            found[signal.key] = min(candidates, key=lambda c: (len(str(c)), str(c)))
    return found


@dataclass
class ChannelReport:
    """1 チャネルの素性(検出結果と実効サンプルレート)。"""

    key: str
    column: str
    spn: int | None
    name_ja: str
    unit: str
    effective_rate_hz: float  # 離散信号では NaN
    n_unique: int
    n_changes: int = 0
    note: str = ""


@dataclass
class LogReport:
    """ログ全体の診断結果。"""

    channels: list[ChannelReport] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    duration_s: float = 0.0
    n_rows: int = 0
    log_rate_hz: float = 0.0

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([c.__dict__ for c in self.channels])

    #: 車体速度として使える(駆動系のねじりを含まない)チャネル
    BODY_SPEED_CHANNELS = ("front_axle_speed_kmh", "speed_kmh")

    def best_speed_channel(self) -> ChannelReport | None:
        """加速度の算出に最も向く車体速度チャネルを返す。

        アウトプットシャフト回転(SPN 191)は 50 Hz で来ることが多いが、
        **駆動系のねじり振動が乗った駆動側の速度**であって車体速度ではない。
        微分すると乗り心地のジャークではなくシャッフル振動を測ってしまうため、
        ここでは候補に含めない。
        """
        rates = {ch.key: ch for ch in self.channels if ch.key in self.BODY_SPEED_CHANNELS}
        if not rates:
            return None
        return max(rates.values(), key=lambda c: (c.effective_rate_hz or 0.0))

    @property
    def jerk_is_reliable(self) -> bool:
        """ジャークを定量指標として扱えるか。

        駆動系のシャッフル振動は 2〜8 Hz。車速の微分でこれを追うには
        エイリアシングを避けて最低 50 Hz(片側 25 Hz)は欲しい。
        通常の J1939 では車速は 10 Hz なので、実質「車体前後加速度センサが
        あるか」で決まる。
        """
        best = self.best_speed_channel()
        return bool(best and best.effective_rate_hz >= 50.0)


def _effective_rate(time: np.ndarray, values: np.ndarray) -> float:
    """値が更新される間隔から実効サンプルレートを推定する。

    CAN のログは全チャネルが同じ時間軸に並ぶことが多いが、実際の更新周期は
    PGN ごとに違う(値が変わらない区間はホールドされている)。
    """
    changed = np.flatnonzero(np.diff(values) != 0)
    if len(changed) < 3:
        return 0.0
    dt = np.median(np.diff(time[changed + 1]))
    return float(1.0 / dt) if dt > 0 else 0.0


def inspect_log(df: pd.DataFrame, mapping: dict[str, str] | None = None) -> LogReport:
    """ログの内容を診断する(検出チャネル・サンプルレート・注意点)。"""

    mapping = mapping or detect_columns(df)
    report = LogReport(n_rows=len(df))

    if "time" not in mapping:
        report.warnings.append("時刻列を検出できませんでした")
        return report

    time = pd.to_numeric(df[mapping["time"]], errors="coerce").to_numpy(dtype=float)
    report.duration_s = float(np.nanmax(time) - np.nanmin(time))
    dt = np.median(np.diff(time))
    report.log_rate_hz = float(1.0 / dt) if dt > 0 else 0.0

    for key, column in mapping.items():
        if key == "time" or key not in CATALOG_BY_KEY:
            continue
        signal = CATALOG_BY_KEY[key]
        values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
        n_changes = int(np.count_nonzero(np.diff(values)))
        note = ""
        if signal.discrete:
            note = f"離散信号({n_changes} 回変化)"
        elif n_changes == 0:
            note = "定数(変化なし)"
        if key in ("gear", "selected_gear"):
            finite = values[np.isfinite(values)]
            if len(finite) and finite.min() >= 120 and finite.max() <= 135:
                note = "生値の可能性(SPN 523/524 は offset -125。125 が N)"
                report.warnings.append(f"{column}: {note}")
        if key == "shift_in_process":
            invalid = float(np.mean((values > 1) & np.isfinite(values)))
            if invalid > 0.01:
                note = f"error/not-available が {invalid:.1%}"
        report.channels.append(
            ChannelReport(
                key=key,
                column=str(column),
                spn=signal.spn,
                name_ja=signal.name_ja,
                unit=signal.unit,
                effective_rate_hz=(
                    float("nan") if signal.discrete
                    else round(_effective_rate(time, values), 1)
                ),
                n_unique=int(pd.Series(values).nunique(dropna=True)),
                n_changes=n_changes,
                note=note,
            )
        )

    report.missing = [
        s.key for s in J1939_CATALOG if s.key not in mapping and s.key != "friction_torque_pct"
    ]
    for key in MINIMUM_KEYS:
        if key not in mapping:
            report.warnings.append(f"必須信号 {key} が見つかりません")
    if "shift_in_process" not in mapping and "gear" not in mapping:
        report.warnings.append(
            "SPN 574(シフトインプロセス)も現在ギヤも無いため、変速イベントを切り出せません"
        )
    best = report.best_speed_channel()
    if not report.jerk_is_reliable:
        rate = f"{best.effective_rate_hz:.0f} Hz ({best.key})" if best else "不明"
        report.warnings.append(
            f"車体速度の実効レートが {rate} しかないため、ジャーク(乗り心地)は"
            "定量指標になりません。定量評価には車体前後加速度(50 Hz 以上)を"
            "追加してください。変速時間・車速の落ち込み・トルク抜けは"
            "この信号セットで問題なく評価できます"
        )
    if any(ch.key == "output_shaft_rpm" for ch in report.channels):
        report.warnings.append(
            "アウトプットシャフト回転(SPN 191)は高レートですが駆動側の速度です。"
            "微分すると駆動系のねじり振動(シャッフル)を測ることになるため、"
            "車体のジャークには使いません(振動の観察用に driveline_speed_kmh として保持)"
        )
    if any(ch.key == "front_axle_speed_kmh" for ch in report.channels):
        report.warnings.append(
            "前輪が非駆動輪であれば SPN 904 は駆動系のねじりを含まないため、"
            "車速(SPN 84)より車体速度の代用に向きます"
        )
    return report


def decode(df: pd.DataFrame, mapping: dict[str, str] | None = None) -> pd.DataFrame:
    """J1939 ログを amtlab の標準チャネル名に変換し、派生量を追加する。

    派生量:
      * ``engine_torque_nm``   = (実トルク率 − フリクション率) × 参照トルク / 100
      * ``demand_torque_nm``   = 要求トルク率 × 参照トルク / 100
      * ``clutch_slip_rpm``    = エンジン回転 − インプットシャフト回転
      * ``gear``               実ギヤ比が 0 付近ならニュートラル(0)に補正
    """

    mapping = mapping or detect_columns(df)
    if "time" not in mapping:
        raise KeyError("時刻列を検出できません。SignalMap で明示してください")

    out = pd.DataFrame()
    for key, column in mapping.items():
        out[key] = pd.to_numeric(df[column], errors="coerce")
    out = out.sort_values("time").reset_index(drop=True)
    out["time"] = out["time"] - out["time"].iloc[0]

    ref = out.get("reference_torque_nm")
    if ref is not None:
        friction = out["friction_torque_pct"] if "friction_torque_pct" in out else 0.0
        if "actual_torque_pct" in out:
            out["engine_torque_nm"] = (out["actual_torque_pct"] - friction) * ref / 100.0
        if "demand_torque_pct" in out:
            out["demand_torque_nm"] = (out["demand_torque_pct"] - friction) * ref / 100.0

    if "engine_speed_rpm" in out and "input_shaft_rpm" in out:
        out["clutch_slip_rpm"] = out["engine_speed_rpm"] - out["input_shaft_rpm"]

    if "accel_pedal_pct" in out:
        out["throttle"] = (out["accel_pedal_pct"] / 100.0).clip(0.0, 1.0)

    # ニュートラル判定: 実ギヤ比が 0 付近 = 動力が切れている
    if "gear_ratio" in out and "gear" in out:
        out.loc[out["gear_ratio"].abs() < 1e-3, "gear"] = 0
    if "gear" in out:
        out["gear"] = out["gear"].fillna(0).round().astype(int)

    if "shift_in_process" in out:
        # 0 = not in process, 1 = in process, 2 = error, 3 = not available
        flag = out["shift_in_process"]
        out["shift_in_process"] = (flag == 1).astype(int).where(flag <= 1, 0)

    # 駆動軸トルクの推定(実測が無い場合の代替)
    if "engine_torque_nm" in out and "gear_ratio" in out:
        out["est_wheel_torque_nm"] = out["engine_torque_nm"] * out["gear_ratio"]

    return out


def format_report(report: LogReport) -> str:
    """診断結果を人が読める文字列にする。"""
    lines = [
        f"rows={report.n_rows}  duration={report.duration_s:.1f}s  "
        f"log rate={report.log_rate_hz:.0f}Hz",
        "",
        f"{'SPN':>5}  {'内部名':<22}{'列名':<34}{'実効Hz':>7}  信号",
    ]
    for ch in sorted(report.channels, key=lambda c: (c.spn or 0)):
        spn = str(ch.spn) if ch.spn else "-"
        note = f"  [{ch.note}]" if ch.note else ""
        rate = "  離散" if np.isnan(ch.effective_rate_hz) else f"{ch.effective_rate_hz:>7.1f}"
        lines.append(
            f"{spn:>5}  {ch.key:<22}{ch.column:<34}{rate:>7}  {ch.name_ja}{note}"
        )
    if report.missing:
        lines += ["", "未検出: " + ", ".join(report.missing)]
    if report.warnings:
        lines += ["", "注意:"] + [f"  - {w}" for w in report.warnings]
    return "\n".join(lines)
