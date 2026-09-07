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
    fig, axes = plt.subplots(5, 1, figsize=(10, 13), sharex=True)

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

    # --- 車速: 落ち込み幅(ピーク->谷)と「変速しなかった場合」との差 ---------
    ax = axes[3]
    _shade_phases(ax, tr)
    sns.lineplot(data=tr, x="time", y="speed_kmh", ax=ax, lw=1.8, color="tab:green",
                 label="vehicle speed")
    win = tr[(tr["time"] >= result.shift_start_time)]
    pre = tr[tr["time"] < result.shift_start_time]
    if len(win) > 2:
        v = win["speed_kmh"].to_numpy()
        t = win["time"].to_numpy()
        i_min = int(np.argmin(v))
        i_peak = int(np.argmax(v[: i_min + 1])) if i_min > 0 else 0
        drop = float(v[i_peak] - v[i_min])
        pre_accel = float(pre["accel"].mean()) if len(pre) else 0.0
        ax.plot(t, v[0] + pre_accel * 3.6 * (t - t[0]), "--", color="0.45", lw=1.2,
                label="no-shift reference")
        loss = float(np.max(v[0] + pre_accel * 3.6 * (t - t[0]) - v))
        if drop > 1e-3:
            ax.annotate(
                "", xy=(t[i_min], v[i_min]), xytext=(t[i_min], v[i_peak]),
                arrowprops={"arrowstyle": "<->", "color": "crimson", "lw": 1.4},
            )
            ax.scatter([t[i_peak], t[i_min]], [v[i_peak], v[i_min]], color="crimson",
                       s=22, zorder=5)
        ax.set_title(
            f"speed drop {drop:.2f} km/h (peak->trough) / shift loss {loss:.2f} km/h "
            "(vs no-shift)", fontsize=9, loc="left",
        )
    ax.set_ylabel("speed [km/h]")
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[4]
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
    kpis = [c for c in ("shift_time_s", "speed_drop_kmh", "speed_loss_kmh",
                        "torque_interrupt_s", "jerk_rms", "jerk_peak") if c in df.columns]
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

    kpis = [c for c in ("shift_time_s", "speed_drop_kmh", "speed_loss_kmh",
                        "torque_interrupt_s", "jerk_rms", "jerk_peak",
                        "clutch_energy_j", "quality_score") if c in df.columns]
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


def plot_pareto(pareto: pd.DataFrame, outdir: Path | str, name: str = "pareto_front",
                keys: tuple[str, ...] | None = None) -> Path:
    """多目的最適化のパレートフロント(第 3 目的は色で表現)。"""
    set_style()
    keys = tuple(keys or OBJECTIVE_KPIS)
    x, y = keys[0], keys[1]
    z = keys[2] if len(keys) > 2 else None
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    if z:
        sc = ax.scatter(pareto[x], pareto[y], c=pareto[z], cmap="viridis", s=70,
                        edgecolor="k", lw=0.4)
        fig.colorbar(sc, ax=ax, label=f"{z}")
    else:
        ax.scatter(pareto[x], pareto[y], s=70, edgecolor="k", lw=0.4)
    best = pareto.sort_values("score").iloc[0]
    ax.scatter([best[x]], [best[y]], marker="*", s=380, color="crimson",
               edgecolor="k", zorder=5, label="selected (weighted best)")
    ax.set_xlabel(f"{x} (lower better)")
    ax.set_ylabel(f"{y} (lower better)")
    ax.set_title("Pareto front: " + " vs ".join(keys))
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


def plot_gear_analysis(df: pd.DataFrame, outdir: Path | str,
                       name: str = "gear_analysis") -> Path:
    """カレントギア(現在ギア)別の KPI 分布。"""
    set_style()
    from .features import gear_long_format

    long = gear_long_format(df)
    g = sns.catplot(
        data=long, x="current_gear", y="value", hue="direction", col="kpi",
        kind="box", col_wrap=3, height=2.9, aspect=1.25, sharey=False,
        showfliers=False, width=0.7,
    )
    g.set_titles("{col_name}")
    g.set_axis_labels("current gear (= from_gear)", "")
    g.figure.suptitle("Shift quality by current gear", y=1.03)
    return _save(g.figure, outdir, name)


def plot_speed_time_relation(df: pd.DataFrame, outdir: Path | str,
                             name: str = "speed_time_relation") -> Path:
    """変速時間と車速の落ち込みの関係(カレントギア別)。"""
    set_style()
    d = df.copy()
    d["current gear"] = d["from_gear"].astype(int)
    d["direction"] = np.where(d["is_upshift"] == 1, "upshift", "downshift")
    metrics = [c for c in ("speed_drop_kmh", "speed_loss_kmh") if c in d.columns]
    long = d.melt(
        id_vars=["shift_time_s", "current gear", "direction", "throttle"],
        value_vars=metrics, var_name="metric", value_name="value",
    )
    g = sns.relplot(
        data=long, x="shift_time_s", y="value", hue="current gear", style="direction",
        col="metric", kind="scatter", palette="viridis", height=3.6, aspect=1.2,
        facet_kws={"sharey": False}, alpha=0.85, s=45,
    )
    for ax, metric in zip(g.axes.flat, metrics):
        sub = d[["shift_time_s", metric]].dropna()
        if len(sub) > 2:
            coef = np.polyfit(sub["shift_time_s"], sub[metric], 1)
            xs = np.linspace(sub["shift_time_s"].min(), sub["shift_time_s"].max(), 20)
            r = float(np.corrcoef(sub["shift_time_s"], sub[metric])[0, 1])
            ax.plot(xs, np.polyval(coef, xs), "--", color="0.35", lw=1.3)
            ax.set_title(f"{metric}  (r = {r:.2f})")
    g.set_axis_labels("shift time [s]", "km/h")
    g.figure.suptitle("Shift time vs speed drop / shift loss", y=1.03)
    return _save(g.figure, outdir, name)


def plot_file_comparison(events: pd.DataFrame, outdir: Path | str,
                         name: str = "file_comparison", max_files: int = 40) -> Path:
    """ログファイル(走行)ごとの KPI ばらつきを比較する。

    ファイル数が多い場合は中央値の順に並べ、上位/下位が見えるようにする。
    """
    set_style()
    kpis = [c for c in ("shift_time_s", "speed_drop_kmh", "speed_loss_kmh",
                        "torque_interrupt_s") if c in events.columns]
    d = events.copy()
    d["file"] = d["source_file"].astype(str)
    order = d.groupby("file")[kpis[0]].median().sort_values().index.tolist()
    if len(order) > max_files:  # 端(良い方/悪い方)を残して間引く
        keep = order[: max_files // 2] + order[-max_files // 2:]
        d = d[d["file"].isin(keep)]
        order = [f for f in order if f in set(keep)]
    long = d.melt(id_vars=["file"], value_vars=kpis, var_name="kpi", value_name="value")
    g = sns.catplot(
        data=long, x="value", y="file", col="kpi", kind="box", order=order,
        col_wrap=2, height=max(3.0, 0.22 * len(order)), aspect=1.5,
        sharex=False, showfliers=False, width=0.7,
    )
    g.set_titles("{col_name}")
    g.set_axis_labels("", "")
    for ax in g.axes.flat:
        ax.tick_params(axis="y", labelsize=7)
    g.figure.suptitle(f"KPI by log file ({len(order)} files)", y=1.01)
    return _save(g.figure, outdir, name)


def plot_signal_presence(presence: pd.DataFrame, outdir: Path | str,
                         name: str = "signal_presence") -> Path:
    """ファイル × 信号の有無マトリクス(どのログに何が入っているか)。"""
    set_style()
    d = presence.set_index("file")
    cols = [c for c in d.columns if d[c].dtype == bool]
    matrix = d[cols].astype(int)
    fig, ax = plt.subplots(
        figsize=(max(7.0, 0.42 * len(cols)), max(3.0, 0.28 * len(matrix)))
    )
    sns.heatmap(matrix, cmap=["#f4c7c3", "#b7e1cd"], cbar=False, linewidths=0.5,
                linecolor="white", ax=ax, vmin=0, vmax=1)
    ax.set_title("Signal availability per log file (green = present)")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelrotation=90, labelsize=8)
    ax.tick_params(axis="y", labelsize=7)
    return _save(fig, outdir, name)


def plot_group_comparison(events: pd.DataFrame, outdir: Path | str,
                          name: str = "group_comparison",
                          group_col: str = "group") -> Path:
    """グループ(A 社 / B 社 など)ごとの KPI 分布。"""
    set_style()
    kpis = [c for c in ("shift_time_s", "speed_drop_kmh", "speed_loss_kmh",
                        "torque_interrupt_s", "jerk_rms") if c in events.columns]
    long = events.melt(id_vars=[group_col], value_vars=kpis,
                       var_name="kpi", value_name="value")
    g = sns.catplot(
        data=long, x=group_col, y="value", hue=group_col, col="kpi",
        kind="violin", col_wrap=3, height=2.9, aspect=1.1, cut=0,
        density_norm="width", sharey=False, legend=False, inner="quartile",
    )
    g.set_titles("{col_name}")
    g.set_axis_labels("", "")
    g.figure.suptitle("KPI distribution by group", y=1.03)
    return _save(g.figure, outdir, name)


def plot_effect_sizes(table: pd.DataFrame, outdir: Path | str,
                      name: str = "effect_sizes") -> Path:
    """基準グループとの差と信頼区間(フォレストプロット)。

    素の差(ファイル単位ブートストラップの 95% CI)と、運転条件を揃えた
    補正後の差を並べて描く。両者が食い違うときは条件の偏りを疑う。
    """
    set_style()
    kpis = list(dict.fromkeys(table["kpi"]))
    n_rows = max(int(table["group"].nunique()), 1)
    height = max(2.6, 1.4 + 0.55 * n_rows)
    fig, axes = plt.subplots(1, len(kpis), figsize=(3.1 * len(kpis), height), squeeze=False)
    for ax, kpi in zip(axes[0], kpis):
        block = table[table["kpi"] == kpi]
        y = np.arange(len(block))
        low = block["diff"] - block["ci_low"]
        high = block["ci_high"] - block["diff"]
        ax.errorbar(block["diff"], y, xerr=[low, high], fmt="o", color="#4c72b0",
                    capsize=4, lw=1.4, markersize=7, label="raw (95% CI)")
        if "adjusted_diff" in block.columns and block["adjusted_diff"].notna().any():
            ax.scatter(block["adjusted_diff"], y, marker="D", color="#dd8452",
                       zorder=5, s=42, label="condition-adjusted")
        ax.axvline(0.0, color="0.35", lw=1.1, ls="--")
        ax.set_yticks(y)
        ax.set_yticklabels(block["group"])
        ax.set_title(kpi, fontsize=10)
        ax.set_xlabel("difference vs reference")
    axes[0][0].legend(fontsize=8, loc="best")
    fig.suptitle("Difference from the reference group (right = worse)", y=1.02)
    fig.tight_layout()
    return _save(fig, outdir, name)


def plot_condition_overlap(events: pd.DataFrame, outdir: Path | str,
                           name: str = "condition_overlap",
                           group_col: str = "group") -> Path:
    """グループごとの運転条件のカバー範囲(比較の公平性チェック)。"""
    set_style()
    d = events.dropna(subset=["speed_kmh"]).copy()
    has_throttle = "throttle" in d.columns and d["throttle"].notna().any()
    g = sns.JointGrid(height=5.5)
    for group, block in d.groupby(group_col, observed=True):
        y = block["throttle"] if has_throttle else block["current_gear"]
        g.ax_joint.scatter(block["speed_kmh"], y, alpha=0.55, s=32, label=str(group))
        sns.kdeplot(x=block["speed_kmh"], ax=g.ax_marg_x, fill=True, alpha=0.35)
        sns.kdeplot(y=y, ax=g.ax_marg_y, fill=True, alpha=0.35)
    g.ax_joint.set_xlabel("vehicle speed [km/h]")
    g.ax_joint.set_ylabel("throttle [-]" if has_throttle else "current gear")
    g.ax_joint.legend(title="group")
    g.figure.suptitle("Operating-condition coverage per group", y=1.01)
    return _save(g.figure, outdir, name)


def plot_group_by_gear(per_gear: pd.DataFrame, outdir: Path | str,
                       name: str = "group_by_gear", kpi: str | None = None) -> Path:
    """カレントギア別の差(層別比較)。"""
    set_style()
    kpi = kpi or per_gear["kpi"].iloc[0]
    block = per_gear[per_gear["kpi"] == kpi]
    g = sns.catplot(
        data=block, x="current_gear", y="diff", hue="group", col="direction",
        kind="bar", height=3.4, aspect=1.3, sharey=True,
    )
    for ax in g.axes.flat:
        ax.axhline(0.0, color="0.35", lw=1.1)
    g.set_axis_labels("current gear", f"{kpi} difference vs reference")
    g.figure.suptitle(f"{kpi}: difference by current gear", y=1.03)
    return _save(g.figure, outdir, name)
