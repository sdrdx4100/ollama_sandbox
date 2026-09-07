"""グループ(仕入先・車両・仕様など)間での変速品質の比較。

「A 社と B 社のログを集めて平均を比べる」は**そのままでは成立しない**。
理由は 2 つある。

1. **運転条件が揃っていない**
   A 社のログが発進加速中心、B 社が高速巡航中心なら、変速時間の差は
   制御の差ではなく走らせ方の差になる。
   → 層別(同じギヤ・同じ負荷どうし)と、条件で補正した差の両方を出す。

2. **イベントが独立でない**
   1 ファイル(1 走行)から複数の変速イベントが出るので、イベント単位で
   検定すると n を過大に見積もる。
   → 信頼区間は**ファイル単位のブートストラップ**で出す。

さらに、信号構成が違うと KPI の定義自体がずれる(SPN 574 の有無で変速時間の
起点が変わる)。これは統計以前の問題なので :func:`check_comparability` で
先に弾く。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

#: 比較対象にする既定の KPI
COMPARISON_KPIS = (
    "shift_time_s",
    "speed_drop_kmh",
    "speed_loss_kmh",
    "torque_interrupt_s",
    "clutch_cycle_time_s",
    "clutch_full_release_ratio",
    "jerk_rms",
)

#: 大きいほど良い KPI(クラッチをしっかり切れている、など)
HIGHER_IS_BETTER = frozenset({"clutch_full_release_ratio"})

#: 良し悪しを一意に決められない KPI(向きだけ report する)
NEUTRAL_KPIS = frozenset(
    {
        "clutch_mean_slip_pct",
        "clutch_peak_slip_pct",
        "clutch_release_rate_pct_s",
        "clutch_engage_rate_pct_s",
        "clutch_slip_integral_pct_s",
    }
)

#: 条件補正に使う説明変数(ログから取れるもの)
CONDITION_FEATURES = ("current_gear", "to_gear", "is_upshift", "speed_kmh", "throttle")


@dataclass
class ComparisonResult:
    """グループ比較の結果。"""

    reference: str
    groups: tuple[str, ...]
    table: pd.DataFrame = field(default_factory=pd.DataFrame)
    per_gear: pd.DataFrame = field(default_factory=pd.DataFrame)
    overlap: pd.DataFrame = field(default_factory=pd.DataFrame)
    comparability: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"基準グループ: {self.reference}"]
        if self.comparability:
            lines.append("")
            lines.append("⚠ 比較可能性の注意:")
            lines += [f"  - {w}" for w in self.comparability]
        if len(self.table):
            lines.append("")
            columns = [
                "kpi", "group", "n", "n_files", "median_ref", "median_group",
                "diff", "diff_pct", "ci_low", "ci_high", "cliffs_delta",
                "adjusted_diff", "verdict",
            ]
            shown = [c for c in columns if c in self.table.columns]
            lines.append(self.table[shown].to_string(index=False))
        return "\n".join(lines)


# ----------------------------------------------------------------------
def check_comparability(
    events: pd.DataFrame, group_col: str = "group", files: pd.DataFrame | None = None
) -> list[str]:
    """グループ間で「そもそも比べてよいか」を確認する。"""

    warnings: list[str] = []
    groups = sorted(events[group_col].dropna().unique())

    # KPI の定義がずれる要因 --------------------------------------------
    if "detection_source" in events.columns:
        per_group = events.groupby(group_col)["detection_source"].agg(
            lambda s: tuple(sorted(set(s)))
        )
        if len(set(per_group)) > 1:
            detail = ", ".join(f"{g}={'/'.join(v)}" for g, v in per_group.items())
            warnings.append(
                "変速イベントの切り出し方法がグループ間で違います "
                f"({detail})。SPN 574 の有無で変速時間の起点が変わるため、"
                "変速時間の直接比較はできません"
            )
        elif any(len(v) > 1 for v in per_group):
            warnings.append(
                "同一グループ内で切り出し方法が混在しています。"
                "detection_source で層別してください"
            )

    if "jerk_quality" in events.columns:
        per_group = events.groupby(group_col)["jerk_quality"].agg(
            lambda s: tuple(sorted(set(s)))
        )
        if len(set(per_group)) > 1:
            warnings.append(
                "ジャークの算出根拠がグループ間で違います "
                f"({', '.join(f'{g}={chr(47).join(v)}' for g, v in per_group.items())})。"
                "jerk_rms の比較は避けてください"
            )

    if "clutch_slip_source" in events.columns:
        per_group = events.groupby(group_col)["clutch_slip_source"].agg(
            lambda s: tuple(sorted(set(s.dropna())))
        )
        if len(set(per_group)) > 1:
            detail = ", ".join(f"{g}={'/'.join(v)}" for g, v in per_group.items())
            warnings.append(
                f"クラッチすべり率の出所がグループ間で違います ({detail})。"
                "SPN 522 の実測と回転差からの代用では、ニュートラル中の"
                "切り深さの見え方が変わるため、クラッチ系 KPI は比較できません"
            )

    # 標本サイズ ----------------------------------------------------------
    counts = events.groupby(group_col).size()
    file_counts = (
        events.groupby(group_col)["source_file"].nunique()
        if "source_file" in events.columns
        else pd.Series(dtype=int)
    )
    for group in groups:
        n_files = int(file_counts.get(group, 0))
        if n_files and n_files < 5:
            warnings.append(
                f"{group}: ログが {n_files} ファイルしかありません。"
                "ファイル単位の信頼区間が広くなります"
            )
        if counts.get(group, 0) < 20:
            warnings.append(f"{group}: 変速イベントが {counts.get(group, 0)} 件と少数です")

    # 運転条件の重なり ----------------------------------------------------
    if len(groups) == 2 and "current_gear" in events.columns:
        gears = {g: set(events.loc[events[group_col] == g, "current_gear"]) for g in groups}
        only_ref = gears[groups[0]] - gears[groups[1]]
        only_other = gears[groups[1]] - gears[groups[0]]
        if only_ref or only_other:
            warnings.append(
                "片方のグループにしか無いカレントギアがあります "
                f"({groups[0]}のみ: {sorted(only_ref)} / {groups[1]}のみ: {sorted(only_other)})"
            )

    if files is not None and len(files) and "error" in files.columns:
        failed = files[files["error"] != ""]
        if len(failed) and group_col in failed.columns:
            per_group = failed.groupby(group_col).size().to_dict()
            warnings.append(f"読み込みに失敗したファイル: {per_group}")
    return warnings


def condition_overlap(
    events: pd.DataFrame,
    group_col: str = "group",
    speed_bins: int = 6,
    throttle_bins: int = 4,
) -> pd.DataFrame:
    """運転条件(ギヤ × 車速 × 負荷)の重なりを集計する。

    両グループに実測があるビンの割合(共通サポート)が低ければ、
    平均の直接比較は意味を持たない。
    """

    d = events.copy()
    if "speed_kmh" in d.columns:
        d["speed_bin"] = pd.qcut(d["speed_kmh"], speed_bins, duplicates="drop")
    if "throttle" in d.columns and d["throttle"].notna().any():
        d["throttle_bin"] = pd.qcut(d["throttle"], throttle_bins, duplicates="drop")
    keys = [k for k in ("current_gear", "speed_bin", "throttle_bin") if k in d.columns]
    if not keys:
        return pd.DataFrame()

    pivot = (
        d.pivot_table(index=keys, columns=group_col, values="shift_time_s",
                      aggfunc="size", observed=True)
        .fillna(0)
        .astype(int)
    )
    pivot["both"] = (pivot > 0).all(axis=1)
    return pivot.reset_index()


def overlap_ratio(overlap: pd.DataFrame, group_col: str = "group") -> float:
    """両グループに実測があるビンが占めるイベントの割合。"""
    if not len(overlap) or "both" not in overlap.columns:
        return float("nan")
    counts = overlap.select_dtypes("number")
    total = counts.to_numpy().sum()
    shared = counts[overlap["both"]].to_numpy().sum()
    return float(shared / total) if total else float("nan")


# ----------------------------------------------------------------------
def hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    """標準化平均差(小標本補正つき)。"""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    pooled = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    if pooled == 0:
        return 0.0
    d = (np.mean(b) - np.mean(a)) / pooled
    correction = 1.0 - 3.0 / (4.0 * (na + nb) - 9.0)
    return float(d * correction)


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """順位ベースの効果量。-1〜+1、正なら b の方が大きい。"""
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return float("nan")
    u = stats.mannwhitneyu(b, a, alternative="two-sided").statistic
    return float(2.0 * u / (na * nb) - 1.0)


def cluster_bootstrap_ci(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    cluster_col: str,
    reference: str,
    other: str,
    n_boot: int = 2000,
    seed: int = 0,
    statistic=np.median,
) -> tuple[float, float]:
    """ファイル(走行)単位でリサンプリングした差の信頼区間。

    同じ走行から出たイベントは相関しているので、イベント単位で
    ブートストラップすると区間が不当に狭くなる。
    """

    rng = np.random.default_rng(seed)
    clusters = {
        group: [
            g[value_col].to_numpy()
            for _, g in df[df[group_col] == group].groupby(cluster_col, observed=True)
        ]
        for group in (reference, other)
    }
    if not clusters[reference] or not clusters[other]:
        return float("nan"), float("nan")

    diffs = np.empty(n_boot)
    for i in range(n_boot):
        sampled = {}
        for group, groups_of_values in clusters.items():
            index = rng.integers(0, len(groups_of_values), len(groups_of_values))
            values = np.concatenate([groups_of_values[j] for j in index])
            sampled[group] = statistic(values) if len(values) else np.nan
        diffs[i] = sampled[other] - sampled[reference]
    finite = diffs[np.isfinite(diffs)]
    if not len(finite):
        return float("nan"), float("nan")
    return float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))


def adjusted_difference(
    events: pd.DataFrame,
    kpi: str,
    group_col: str,
    reference: str,
    other: str,
    seed: int = 0,
) -> float:
    """運転条件を揃えたときのグループ差(標準化 / G-computation)。

    「条件 + グループ」から KPI を学習し、**同じ条件の集合**に対して
    グループだけを入れ替えて予測した差の平均をとる。運転のさせ方の違いを
    取り除いた、制御そのものの差の推定値。
    """

    from sklearn.ensemble import HistGradientBoostingRegressor

    features = [c for c in CONDITION_FEATURES if c in events.columns]
    subset = events[events[group_col].isin([reference, other])]
    subset = subset[subset[kpi].notna()]
    if len(subset) < 20 or not features:
        return float("nan")

    x = subset[features].astype(float).copy()
    x["__group"] = (subset[group_col] == other).astype(float).to_numpy()
    y = subset[kpi].astype(float).to_numpy()

    model = HistGradientBoostingRegressor(
        random_state=seed, max_iter=200, min_samples_leaf=10
    )
    model.fit(x, y)

    as_ref = x.copy()
    as_ref["__group"] = 0.0
    as_other = x.copy()
    as_other["__group"] = 1.0
    return float(np.mean(model.predict(as_other) - model.predict(as_ref)))


def _verdict(diff: float, ci_low: float, ci_high: float, delta: float, kpi: str) -> str:
    """差の読み方を一言で。KPI ごとに「大きいほど良い」向きを考慮する。"""
    if not np.isfinite(ci_low) or not np.isfinite(ci_high):
        return "判定不能"
    if ci_low <= 0.0 <= ci_high:
        return "差は有意でない"
    size = "大" if abs(delta) >= 0.474 else "中" if abs(delta) >= 0.33 else "小"
    if kpi in NEUTRAL_KPIS:
        return f"{'増加' if diff > 0 else '減少'}(効果量: {size})"
    worse = diff > 0 if kpi not in HIGHER_IS_BETTER else diff < 0
    return f"{'悪化' if worse else '改善'}(効果量: {size})"


def compare_groups(
    events: pd.DataFrame,
    reference: str | None = None,
    kpis: tuple[str, ...] = COMPARISON_KPIS,
    group_col: str = "group",
    cluster_col: str = "source_file",
    n_boot: int = 1000,
    seed: int = 0,
    adjust: bool = True,
    files: pd.DataFrame | None = None,
) -> ComparisonResult:
    """グループ間で KPI を比較する(基準グループとの差)。"""

    if group_col not in events.columns:
        raise KeyError(f"グループ列 {group_col!r} がありません")
    groups = [g for g in events[group_col].dropna().unique()]
    if len(groups) < 2:
        raise ValueError(f"比較には 2 グループ以上必要です: {groups}")
    groups = sorted(groups, key=str)
    reference = reference or groups[0]
    if reference not in groups:
        raise ValueError(f"基準グループ {reference!r} がデータにありません")
    others = [g for g in groups if g != reference]

    if cluster_col not in events.columns:
        events = events.copy()
        events[cluster_col] = "__all__"

    rows = []
    for kpi in [k for k in kpis if k in events.columns]:
        ref_values = events.loc[events[group_col] == reference, kpi].dropna().to_numpy()
        for group in others:
            values = events.loc[events[group_col] == group, kpi].dropna().to_numpy()
            if not len(ref_values) or not len(values):
                continue
            median_ref = float(np.median(ref_values))
            median_group = float(np.median(values))
            diff = median_group - median_ref
            ci_low, ci_high = cluster_bootstrap_ci(
                events, kpi, group_col, cluster_col, reference, group, n_boot, seed
            )
            delta = cliffs_delta(ref_values, values)
            rows.append(
                {
                    "kpi": kpi,
                    "group": group,
                    "n_ref": len(ref_values),
                    "n": len(values),
                    "n_files_ref": int(
                        events.loc[events[group_col] == reference, cluster_col].nunique()
                    ),
                    "n_files": int(
                        events.loc[events[group_col] == group, cluster_col].nunique()
                    ),
                    "median_ref": round(median_ref, 4),
                    "median_group": round(median_group, 4),
                    "diff": round(diff, 4),
                    "diff_pct": round(diff / median_ref * 100.0, 1) if median_ref else np.nan,
                    "ci_low": round(ci_low, 4),
                    "ci_high": round(ci_high, 4),
                    "hedges_g": round(hedges_g(ref_values, values), 3),
                    "cliffs_delta": round(delta, 3),
                    "adjusted_diff": (
                        round(
                            adjusted_difference(events, kpi, group_col, reference, group, seed),
                            4,
                        )
                        if adjust
                        else np.nan
                    ),
                    "verdict": _verdict(diff, ci_low, ci_high, delta, kpi),
                }
            )

    table = pd.DataFrame(rows)
    overlap = condition_overlap(events, group_col)
    result = ComparisonResult(
        reference=reference,
        groups=tuple(groups),
        table=table,
        per_gear=compare_by_gear(events, reference, kpis, group_col),
        overlap=overlap,
        comparability=check_comparability(events, group_col, files),
    )
    ratio = overlap_ratio(overlap, group_col)
    if np.isfinite(ratio) and ratio < 0.7:
        result.comparability.append(
            f"運転条件の共通サポートが {ratio:.0%} しかありません。"
            "層別(per_gear)と条件補正後の差(adjusted_diff)を優先してください"
        )
    return result


def compare_by_gear(
    events: pd.DataFrame,
    reference: str,
    kpis: tuple[str, ...] = COMPARISON_KPIS,
    group_col: str = "group",
) -> pd.DataFrame:
    """カレントギア別・変速方向別の層別比較(中央値の差)。"""

    if "current_gear" not in events.columns:
        return pd.DataFrame()
    d = events.copy()
    d["direction"] = np.where(d.get("is_upshift", 1) == 1, "upshift", "downshift")
    rows = []
    columns = [k for k in kpis if k in d.columns]
    for (gear, direction), block in d.groupby(["current_gear", "direction"], observed=True):
        if block[group_col].nunique() < 2 or reference not in set(block[group_col]):
            continue
        for kpi in columns:
            ref_values = block.loc[block[group_col] == reference, kpi].dropna()
            for group, values in block.groupby(group_col, observed=True)[kpi]:
                values = values.dropna()
                if group == reference or len(values) < 3 or len(ref_values) < 3:
                    continue
                rows.append(
                    {
                        "current_gear": int(gear),
                        "direction": direction,
                        "kpi": kpi,
                        "group": group,
                        "n_ref": len(ref_values),
                        "n": len(values),
                        "median_ref": round(float(ref_values.median()), 4),
                        "median_group": round(float(values.median()), 4),
                        "diff": round(float(values.median() - ref_values.median()), 4),
                    }
                )
    return pd.DataFrame(rows)
