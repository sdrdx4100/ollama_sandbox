"""seaborn による解析結果の可視化。

図中のラベルはフォント依存を避けるため英語表記で統一する
(日本語フォントが利用可能な場合は ``set_style`` が自動で登録する)。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from .features import OBJECTIVE_KPIS  # noqa: E402
from .simulation.plant import ShiftResult  # noqa: E402

JP_FONT_CANDIDATES = (
    "Noto Sans CJK JP",
    "IPAexGothic",
    "IPAGothic",
    "Hiragino Sans",
    "Yu Gothic",
    "TakaoGothic",
)

PHASE_COLORS = {
    "in_gear": "#f7f7f7",
    "torque_reduce": "#fde0c5",
    "clutch_open": "#f9b58a",
    "gear_out": "#f08f6e",
    "speed_sync": "#c8d9ee",
    "gear_in": "#9dbdd9",
    "clutch_close": "#a8ddb5",
    "torque_recovery": "#d9f0a3",
    "done": "#ffffff",
}


def set_style(context: str = "notebook") -> None:
    """seaborn の共通スタイルを設定する。"""
    sns.set_theme(style="whitegrid", context=context, palette="deep")
    available = {f.name for f in fm.fontManager.ttflist}
    for name in JP_FONT_CANDIDATES:
        if name in available:
            plt.rcParams["font.family"] = [name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["savefig.bbox"] = "tight"


def _save(fig: plt.Figure, outdir: Path | str, name: str) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def _shade_phases(ax: plt.Axes, trace: pd.DataFrame) -> None:
    phase = trace["phase"].to_numpy()
    time = trace["time"].to_numpy()
    start = 0
    for i in range(1, len(phase) + 1):
        if i == len(phase) or phase[i] != phase[start]:
            ax.axvspan(
                time[start], time[i - 1],
                color=PHASE_COLORS.get(phase[start], "#ffffff"), alpha=0.55, lw=0,
            )
            start = i


def plot_shift_trace(result: ShiftResult, outdir: Path | str, name: str = "shift_trace") -> Path:
    """1 変速イベントの時系列(回転・トルク・クラッチ・加速度/ジャーク)。"""
    set_style()
    tr = result.trace
    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)

    ax = axes[0]
    _shade_phases(ax, tr)
    sns.lineplot(data=tr, x="time", y="engine_speed_rpm", ax=ax, label="engine", lw=1.6)
    sns.lineplot(data=tr, x="time", y="input_speed_rpm", ax=ax, label="input shaft", lw=1.6)
    ax.set_ylabel("speed [rpm]")
    sc = result.scenario
    ax.set_title(
        f"{sc.from_gear}->{sc.to_gear} shift @ {sc.speed_kmh:.0f} km/h, "
        f"throttle {sc.throttle:.0%}, grade {sc.grade_pct:.1f}%"
    )

    ax = axes[1]
    _shade_phases(ax, tr)
    sns.lineplot(data=tr, x="time", y="engine_torque", ax=ax, label="engine torque", lw=1.6)
    sns.lineplot(data=tr, x="time", y="clutch_torque", ax=ax, label="clutch torque", lw=1.4)
    sns.lineplot(data=tr, x="time", y="shaft_torque", ax=ax, label="driveshaft torque", lw=1.4)
    ax.set_ylabel("torque [Nm]")

    ax = axes[2]
    _shade_phases(ax, tr)
    sns.lineplot(data=tr, x="time", y="clutch_position", ax=ax, label="clutch position", lw=1.6)
    ax2 = ax.twinx()
    ax2.step(tr["time"], tr["gear"], color="0.35", lw=1.2, where="post")
    ax2.set_ylabel("gear (0 = neutral)")
    ax2.grid(False)
    ax.set_ylabel("clutch stroke [-]")

    ax = axes[3]
    _shade_phases(ax, tr)
    sns.lineplot(data=tr, x="time", y="accel_filt", ax=ax, label="accel [m/s2]", lw=1.6)
    ax2 = ax.twinx()
    ax2.plot(tr["time"], tr["jerk"], color="crimson", lw=1.0, alpha=0.8, label="jerk")
    ax2.set_ylabel("jerk [m/s3]")
    ax2.grid(False)
    ax.set_ylabel("accel [m/s2]")
    ax.set_xlabel("time [s]")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.55)
        for p, c in PHASE_COLORS.items() if p not in ("in_gear", "done")
    ]
    labels = [p for p in PHASE_COLORS if p not in ("in_gear", "done")]
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return _save(fig, outdir, name)


def plot_trace_comparison(
    results: dict[str, ShiftResult], outdir: Path | str, name: str = "trace_comparison"
) -> Path:
    """適合値違いの時系列比較(加速度・ジャーク・エンジン回転)。"""
    set_style()
    frames = []
    for label, res in results.items():
        tr = res.trace[["time", "accel_filt", "jerk", "engine_speed_rpm", "shaft_torque"]].copy()
        tr["calibration"] = label
        frames.append(tr)
    df = pd.concat(frames, ignore_index=True)
    long = df.melt(
        id_vars=["time", "calibration"],
        value_vars=["engine_speed_rpm", "shaft_torque", "accel_filt", "jerk"],
        var_name="signal", value_name="value",
    )
    g = sns.relplot(
        data=long, x="time", y="value", hue="calibration", row="signal",
        kind="line", height=2.2, aspect=4.0, facet_kws={"sharey": False}, lw=1.4,
    )
    g.set_titles(row_template="{row_name}")
    g.set_axis_labels("time [s]", "")
    g.figure.suptitle("Shift response: baseline vs optimized", y=1.01)
    return _save(g.figure, outdir, name)


def plot_kpi_distributions(df: pd.DataFrame, outdir: Path | str,
                           name: str = "kpi_distributions") -> Path:
    """DoE データセットの KPI 分布(アップ/ダウンシフト別)。"""
    set_style()
    kpis = [c for c in ("shift_time_s", "jerk_rms", "jerk_peak", "clutch_energy_j",
                        "torque_interrupt_s", "engine_flare_rpm") if c in df.columns]
    long = df.melt(id_vars=["is_upshift"], value_vars=kpis,
                   var_name="kpi", value_name="value")
    long["direction"] = np.where(long["is_upshift"] == 1, "upshift", "downshift")
    g = sns.catplot(
        data=long, x="direction", y="value", hue="direction", col="kpi",
        kind="violin", col_wrap=3, height=2.8, aspect=1.15, cut=0,
        density_norm="width", sharey=False, legend=False,
    )
    g.set_titles("{col_name}")
    g.set_axis_labels("", "")
    g.figure.suptitle("Shift quality KPI distribution (random DoE)", y=1.02)
    return _save(g.figure, outdir, name)


def plot_kpi_correlation(df: pd.DataFrame, outdir: Path | str,
                         name: str = "kpi_correlation") -> Path:
    """適合値・条件と KPI の相関ヒートマップ(Spearman)。"""
    set_style()
    from .dataset import FEATURE_COLUMNS

    kpis = [c for c in ("shift_time_s", "jerk_rms", "jerk_peak", "clutch_energy_j",
                        "torque_interrupt_s", "engine_flare_rpm", "quality_score")
            if c in df.columns]
    cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    corr = df[cols + kpis].corr(method="spearman").loc[cols, kpis]
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax,
                cbar_kws={"label": "Spearman rho"})
    ax.set_title("Correlation: inputs vs shift-quality KPIs")
    return _save(fig, outdir, name)


def plot_condition_map(df: pd.DataFrame, outdir: Path | str, kpi: str = "jerk_rms",
                       name: str = "condition_map") -> Path:
    """車速 x スロットルの運転領域における KPI マップ。"""
    set_style()
    d = df.copy()
    d["speed_bin"] = pd.cut(d["speed_kmh"], bins=np.arange(0, 201, 25))
    d["throttle_bin"] = pd.cut(d["throttle"], bins=np.linspace(0, 1, 6))
    pivot = d.pivot_table(index="throttle_bin", columns="speed_bin",
                          values=kpi, aggfunc="mean", observed=False)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="rocket_r", ax=ax,
                cbar_kws={"label": kpi})
    ax.invert_yaxis()
    ax.set_title(f"Operating-region map of {kpi}")
    ax.set_xlabel("vehicle speed [km/h]")
    ax.set_ylabel("throttle [-]")
    return _save(fig, outdir, name)


def plot_model_diagnostics(surrogate, outdir: Path | str,
                           name: str = "model_diagnostics") -> Path:
    """代理モデルの予測精度(OOF 予測 vs 実測、残差分布)。"""
    set_style()
    preds = surrogate.cv_predictions.copy()
    preds["direction"] = np.where(preds.get("is_upshift", 1) == 1, "upshift", "downshift")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    sns.scatterplot(data=preds, x="actual", y="predicted", hue="direction",
                    alpha=0.75, ax=axes[0])
    lo = float(min(preds["actual"].min(), preds["predicted"].min()))
    hi = float(max(preds["actual"].max(), preds["predicted"].max()))
    axes[0].plot([lo, hi], [lo, hi], "k--", lw=1)
    m = surrogate.metrics
    axes[0].set_title(
        f"{surrogate.target}: OOF prediction "
        f"(R2={m['r2']:.3f}, RMSE={m['rmse']:.3f})"
    )
    sns.histplot(data=preds, x="residual", hue="direction", kde=True, ax=axes[1])
    axes[1].set_title("Residual distribution")
    fig.tight_layout()
    return _save(fig, outdir, name)


def plot_importance(importance: pd.DataFrame, outdir: Path | str,
                    name: str = "importance") -> Path:
    """permutation importance の棒グラフ。"""
    set_style()
    df = importance.head(14)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sns.barplot(data=df, y="feature", x="importance", hue="kind", dodge=False, ax=ax)
    ax.errorbar(df["importance"], range(len(df)), xerr=df["std"], fmt="none",
                ecolor="0.3", capsize=3, lw=1)
    target = df["target"].iloc[0] if "target" in df.columns else ""
    ax.set_title(f"Permutation importance ({target})")
    ax.set_xlabel("R2 decrease")
    ax.set_ylabel("")
    return _save(fig, outdir, name)


def plot_partial_dependence(pdp: pd.DataFrame, outdir: Path | str,
                            name: str = "partial_dependence") -> Path:
    """適合値ごとの部分依存プロット。"""
    set_style()
    g = sns.relplot(data=pdp, x="value", y="partial_dependence", col="feature",
                    col_wrap=4, kind="line", height=2.4, aspect=1.15,
                    facet_kws={"sharex": False, "sharey": True}, lw=1.8)
    g.set_titles("{col_name}")
    target = pdp["target"].iloc[0] if "target" in pdp.columns else ""
    g.set_axis_labels("calibration value", target)
    g.figure.suptitle(f"Partial dependence of {target} on calibration parameters", y=1.03)
    return _save(g.figure, outdir, name)


def plot_optuna_history(trials: pd.DataFrame, outdir: Path | str,
                        name: str = "optuna_history") -> Path:
    """Optuna 試行履歴(単目的)。"""
    set_style()
    fig, ax = plt.subplots(figsize=(9, 4.2))
    sns.scatterplot(data=trials, x="trial", y="objective", ax=ax, alpha=0.65,
                    label="trial")
    if "best_so_far" in trials.columns:
        sns.lineplot(data=trials, x="trial", y="best_so_far", ax=ax, color="crimson",
                     lw=2, label="best so far")
    ax.set_title("Optuna optimization history")
    ax.set_ylabel("objective (lower is better)")
    return _save(fig, outdir, name)


def plot_pareto(pareto: pd.DataFrame, outdir: Path | str, name: str = "pareto_front") -> Path:
    """多目的最適化のパレートフロント。"""
    set_style()
    x, y, z = OBJECTIVE_KPIS
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    sc = ax.scatter(pareto[x], pareto[y], c=pareto[z], cmap="viridis", s=70,
                    edgecolor="k", lw=0.4)
    best = pareto.sort_values("score").iloc[0]
    ax.scatter([best[x]], [best[y]], marker="*", s=380, color="crimson",
               edgecolor="k", zorder=5, label="selected (weighted best)")
    fig.colorbar(sc, ax=ax, label=f"{z}")
    ax.set_xlabel(f"{x} (lower better)")
    ax.set_ylabel(f"{y} (lower better)")
    ax.set_title("Pareto front: comfort vs shift time vs clutch energy")
    ax.legend()
    return _save(fig, outdir, name)


def plot_calibration_comparison(compare: pd.DataFrame, outdir: Path | str,
                                name: str = "calibration_comparison") -> Path:
    """ベースラインと最適化後の KPI を代表条件ごとに比較する。"""
    set_style()
    d = compare.copy()
    d["case"] = (
        d["from_gear"].astype(int).astype(str) + "->" + d["to_gear"].astype(int).astype(str)
        + "\n" + d["speed_kmh"].round(0).astype(int).astype(str) + "km/h"
    )
    kpis = ["jerk_rms", "jerk_peak", "shift_time_s", "clutch_energy_j"]
    long = d.melt(id_vars=["case", "calibration"], value_vars=kpis,
                  var_name="kpi", value_name="value")
    g = sns.catplot(data=long, x="case", y="value", hue="calibration", col="kpi",
                    kind="bar", col_wrap=2, height=3.0, aspect=1.7, sharey=False)
    g.set_titles("{col_name}")
    g.set_axis_labels("", "")
    for ax in g.axes.flat:
        ax.tick_params(axis="x", labelsize=8)
    g.figure.suptitle("Baseline vs optimized calibration", y=1.02)
    return _save(g.figure, outdir, name)


def plot_log_overview(trace: pd.DataFrame, events: pd.DataFrame, outdir: Path | str,
                      name: str = "log_overview") -> Path:
    """実ログ全体の俯瞰図(回転・車速・ギヤ・ジャークと検出イベント)。"""
    set_style()
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)

    spans = []
    if {"shift_time_s"}.issubset(events.columns) and "t_start" in events.columns:
        spans = list(zip(events["t_start"], events["t_start"] + events["shift_time_s"]))

    for ax in axes:
        for t0, t1 in spans:
            ax.axvspan(t0, t1, color="#f08f6e", alpha=0.25, lw=0)

    sns.lineplot(data=trace, x="time", y="engine_speed_rpm", ax=axes[0], lw=1.3)
    axes[0].set_ylabel("engine [rpm]")
    axes[0].set_title(f"Vehicle log overview ({len(events)} shift events detected)")

    sns.lineplot(data=trace, x="time", y="speed_kmh", ax=axes[1], lw=1.3, color="tab:green")
    ax2 = axes[1].twinx()
    ax2.step(trace["time"], trace["gear"], where="post", color="0.35", lw=1.1)
    ax2.set_ylabel("gear")
    ax2.grid(False)
    axes[1].set_ylabel("speed [km/h]")

    sns.lineplot(data=trace, x="time", y="jerk", ax=axes[2], lw=0.9, color="crimson")
    axes[2].set_ylabel("jerk [m/s3]")
    axes[2].set_xlabel("time [s]")
    fig.tight_layout()
    return _save(fig, outdir, name)
