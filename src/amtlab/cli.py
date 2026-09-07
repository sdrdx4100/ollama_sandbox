"""コマンドラインインタフェース。

例:
    amtlab run --config configs/default.yaml
    amtlab simulate --scenario 2-3@45,0.7 --out outputs/figures
    amtlab dataset --samples 800 --out outputs/data/doe.csv
    amtlab analyze --dataset outputs/data/doe.csv
    amtlab calibrate --trials 200 --mode pareto
    amtlab report --summary outputs/summary.json
    amtlab ollama
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", type=str, default=None, help="設定 YAML のパス")
    p.add_argument("--output", type=str, default=None, help="出力ディレクトリ(設定を上書き)")


def cmd_run(args: argparse.Namespace) -> int:
    from .config import PipelineConfig
    from .pipeline import run_pipeline

    cfg = PipelineConfig.load(args.config)
    if args.output:
        cfg.output_dir = args.output
    if args.samples:
        cfg.dataset.n_samples = args.samples
    if args.model_trials:
        cfg.model.n_trials = args.model_trials
    if args.calib_trials:
        cfg.calibration.n_trials = args.calib_trials
    if args.mode:
        cfg.calibration.mode = args.mode
    if args.no_ollama:
        cfg.ollama.enabled = False
    if args.quick:
        cfg.dataset.n_samples = min(cfg.dataset.n_samples, 120)
        cfg.model.n_trials = min(cfg.model.n_trials, 10)
        cfg.calibration.n_trials = min(cfg.calibration.n_trials, 30)
    run_pipeline(cfg)
    return 0


def cmd_simulate(args: argparse.Namespace) -> int:
    from . import viz
    from .features import extract_kpis
    from .pipeline import load_control, quick_scenario
    from .simulation.plant import simulate_shift

    scenario = quick_scenario(args.scenario)
    control = load_control(args.control)
    result = simulate_shift(control, scenario)
    kpis = extract_kpis(result)
    print(json.dumps({k: round(float(v), 4) for k, v in kpis.items()},
                     ensure_ascii=False, indent=2))
    outdir = Path(args.out)
    if args.trace_csv:
        outdir.mkdir(parents=True, exist_ok=True)
        result.trace.to_csv(outdir / "trace.csv", index=False)
    path = viz.plot_shift_trace(result, outdir, args.name)
    print(f"figure: {path}")
    return 0


def cmd_dataset(args: argparse.Namespace) -> int:
    from .dataset import build_dataset

    df = build_dataset(n_samples=args.samples, seed=args.seed, n_jobs=args.jobs)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"{len(df)} events -> {out}")
    print(df[["shift_time_s", "jerk_rms", "clutch_energy_j"]].describe().round(3).to_string())
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    from .config import PipelineConfig
    from .pipeline import run_pipeline

    cfg = PipelineConfig.load(args.config)
    if args.output:
        cfg.output_dir = args.output
    if args.no_ollama:
        cfg.ollama.enabled = False
    df = pd.read_csv(args.dataset)
    run_pipeline(cfg, dataset=df)
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    from . import viz
    from .calibration import CalibrationSetting, calibrate
    from .config import PipelineConfig

    cfg = PipelineConfig.load(args.config)
    setting = CalibrationSetting(
        objectives=cfg.calibration.objective_spec(),
        worst_case_weight=cfg.calibration.worst_case_weight,
    )
    outcome = calibrate(
        setting=setting,
        n_trials=args.trials or cfg.calibration.n_trials,
        mode=args.mode or cfg.calibration.mode,
        seed=cfg.calibration.seed,
    )
    out = Path(args.output or cfg.output_dir)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    outcome.improvement().to_csv(out / "tables" / "improvement.csv", index=False)
    print(outcome.improvement().round(3).to_string(index=False))
    print("\n最適化された適合値:")
    print(json.dumps({k: round(v, 3) for k, v in outcome.best_control.as_dict().items()},
                     ensure_ascii=False, indent=2))
    if len(outcome.pareto):
        outcome.pareto.to_csv(out / "tables" / "pareto_front.csv", index=False)
        fig = viz.plot_pareto(outcome.pareto, out / "figures", keys=setting.objectives.keys)
        print(f"figure: {fig}")
    return 0


def _parse_groups(specs: list[str]) -> dict[str, str]:
    """``"A社=logs/a"`` の並びを {グループ: パターン} にする。"""
    groups: dict[str, str] = {}
    for spec in specs:
        name, sep, pattern = spec.partition("=")
        if not sep or not pattern:
            raise SystemExit(f"--group は 名前=パス の形式で指定してください: {spec!r}")
        if name.strip() in groups:
            raise SystemExit(f"グループ名が重複しています: {name.strip()!r}")
        groups[name.strip()] = pattern.strip()
    return groups


def cmd_compare(args: argparse.Namespace) -> int:
    """グループ(A 社 / B 社 など)ごとにログを集めて比較する。"""
    from . import viz
    from .batch import analyze_logs, signal_presence
    from .compare import compare_groups
    from .config import PipelineConfig
    from .reporting import (
        OllamaClient,
        build_comparison_summary,
        generate_comparison_report,
        write_report,
    )

    groups = _parse_groups(args.group)
    if len(groups) < 2:
        raise SystemExit("--group は 2 つ以上指定してください")

    sm = _signal_map_from_args(args)
    batch = analyze_logs(groups, sm, resample_hz=args.resample_hz, n_jobs=args.jobs)
    print(batch.summary())
    if batch.events.empty:
        print("\n変速イベントを検出できませんでした")
        return 1

    out = Path(args.out)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    batch.events.to_csv(out / "tables" / "group_events.csv", index=False)
    batch.files.to_csv(out / "tables" / "group_files.csv", index=False)

    comparison = compare_groups(
        batch.events, reference=args.reference, files=batch.files,
        n_boot=args.boot, seed=args.seed,
    )
    comparison.table.to_csv(out / "tables" / "group_comparison.csv", index=False)
    if len(comparison.per_gear):
        comparison.per_gear.to_csv(out / "tables" / "group_per_gear.csv", index=False)
    if len(comparison.overlap):
        comparison.overlap.to_csv(out / "tables" / "condition_overlap.csv", index=False)
    signal_presence(groups, sm).to_csv(out / "tables" / "signal_presence.csv", index=False)

    print()
    print(comparison.summary())

    figures = [
        viz.plot_group_comparison(batch.events, out / "figures"),
        viz.plot_effect_sizes(comparison.table, out / "figures"),
        viz.plot_condition_overlap(batch.events, out / "figures"),
    ]
    if len(comparison.per_gear):
        figures.append(viz.plot_group_by_gear(comparison.per_gear, out / "figures"))
    print()
    for path in figures:
        print(f"figure: {path}")

    cfg = PipelineConfig.load(args.config)
    client = OllamaClient(host=cfg.ollama.host, model=cfg.ollama.model,
                          temperature=cfg.ollama.temperature, timeout_s=cfg.ollama.timeout_s)
    summary = build_comparison_summary(comparison, batch)
    (out / "comparison_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    text, source = generate_comparison_report(
        summary, client=client, enabled=cfg.ollama.enabled and not args.no_ollama
    )
    report = write_report(text, out / "comparison_report.md")
    print(f"report: {report} (source: {source})")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from .config import PipelineConfig
    from .reporting import OllamaClient, generate_report, write_report

    cfg = PipelineConfig.load(args.config)
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    client = OllamaClient(host=cfg.ollama.host, model=args.model or cfg.ollama.model,
                          temperature=cfg.ollama.temperature, timeout_s=cfg.ollama.timeout_s)
    text, source = generate_report(summary, client=client, enabled=not args.no_ollama,
                                   language=cfg.ollama.language)
    path = write_report(text, args.out)
    print(f"{path} (source: {source})")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    from . import viz
    from .batch import analyze_logs, iter_log_paths, signal_presence
    from .features import gear_summary
    from .ingest import analyze_log

    sm = _signal_map_from_args(args)
    paths = iter_log_paths(args.log)
    if not paths:
        print(f"ログが見つかりません: {args.log}")
        return 1

    out = Path(args.out)
    (out / "tables").mkdir(parents=True, exist_ok=True)

    # --- 単一ファイル: 従来どおり時系列も描く ----------------------------
    if len(paths) == 1:
        trace, kpi = analyze_log(paths[0], sm, resample_hz=args.resample_hz)
        kpi.to_csv(out / "tables" / "log_events.csv", index=False)
        if kpi.empty:
            print("変速イベントを検出できませんでした(ギヤ信号を確認してください)")
            return 1
        print(f"検出: {len(trace.attrs.get('mapping', {}))} 信号 / "
              f"加速度の出所: {trace.attrs.get('accel_source')} "
              f"(jerk 信頼度: {trace.attrs.get('jerk_quality')})")
        print(kpi.round(3).to_string(index=False))
        print(f"figure: {viz.plot_log_overview(trace, kpi, out / 'figures')}")
        figure = viz.plot_kpi_distributions(
            kpi, out / "figures", "log_kpi_distributions"
        )
        print(f"figure: {figure}")
        return 0

    # --- 複数ファイル: KPI 表に畳んで比較 --------------------------------
    result = analyze_logs(
        paths, sm, resample_hz=args.resample_hz, n_jobs=args.jobs
    )
    result.files.to_csv(out / "tables" / "log_files.csv", index=False)
    print(result.summary())
    if result.events.empty:
        print("\n変速イベントを検出できませんでした")
        return 1

    result.events.to_csv(out / "tables" / "log_events.csv", index=False)
    gears = gear_summary(result.events)
    gears.to_csv(out / "tables" / "log_gear_summary.csv", index=False)
    presence = signal_presence(paths, sm)
    presence.to_csv(out / "tables" / "signal_presence.csv", index=False)

    print("\nカレントギア別:")
    columns = [c for c in ("current_gear", "direction", "shift_time_s_count",
                           "shift_time_s_mean", "speed_drop_kmh_mean",
                           "speed_loss_kmh_mean") if c in gears.columns]
    print(gears[columns].to_string(index=False))

    figures = [
        viz.plot_gear_analysis(result.events, out / "figures", "log_gear_analysis"),
        viz.plot_file_comparison(result.events, out / "figures"),
        viz.plot_signal_presence(presence, out / "figures"),
        viz.plot_kpi_distributions(result.events, out / "figures",
                                   "log_kpi_distributions"),
    ]
    for path in figures:
        print(f"figure: {path}")
    return 0


def _signal_map_from_args(args: argparse.Namespace):
    from .ingest import SignalMap

    if getattr(args, "map", None):
        sm = SignalMap.from_file(args.map)
        sm.auto_detect = not getattr(args, "no_auto", False)
        return sm
    return SignalMap(
        time=args.col_time,
        engine_speed_rpm=args.col_engine_speed,
        speed_kmh=args.col_speed,
        gear=args.col_gear,
        throttle=args.col_throttle,
        shaft_torque=args.col_torque,
        auto_detect=not args.no_auto,
    )


def cmd_inspect(args: argparse.Namespace) -> int:
    """ログの信号構成とサンプルレートを診断する(複数ファイル可)。"""
    from .batch import iter_log_paths, read_log, signal_presence
    from .ingest import SignalMap
    from .j1939 import format_report, inspect_log

    paths = iter_log_paths(args.log)
    if not paths:
        print(f"ログが見つかりません: {args.log}")
        return 1

    if len(paths) > 1:
        presence = signal_presence(paths, _signal_map_from_args(args)
                                   if args.map else SignalMap())
        flags = [c for c in presence.columns if presence[c].dtype == bool]
        broken = presence[presence["error"] != ""]
        ok = presence[presence["error"] == ""]
        print(f"{len(paths)} ファイル(読み込み可 {len(ok)} / 失敗 {len(broken)})")

        everywhere = [c for c in flags if len(ok) and ok[c].all()]
        partial = [c for c in flags if len(ok) and not ok[c].all() and ok[c].any()]
        nowhere = [c for c in flags if len(ok) and not ok[c].any()]
        print(f"\n全ファイルにある信号 ({len(everywhere)}): " + ", ".join(everywhere))
        if nowhere:
            print(f"どのファイルにも無い信号 ({len(nowhere)}): " + ", ".join(nowhere))
        if partial:
            print("\n一部のファイルにしか無い信号:")
            for column in partial:
                missing = ok.loc[~ok[column], "file"].tolist()
                shown = ", ".join(missing[:5]) + (" ..." if len(missing) > 5 else "")
                print(f"  {column}: {len(missing)} ファイルで欠落 ({shown})")
        for _, row in broken.iterrows():
            print(f"  [読み込み失敗] {row['file']}: {row['error']}")
        if args.save:
            presence_path = Path(args.save).with_name(
                Path(args.save).stem + "_presence.csv"
            )
            presence.to_csv(presence_path, index=False)
            print(f"\n-> {presence_path}")

        readable = [p for p in paths if p.name in set(ok["file"])]
        if not readable:
            return 1
        paths = readable[:1]
        print(f"\n代表 1 ファイル ({paths[0].name}) の詳細:")

    df = read_log(paths[0])
    mapping = SignalMap.from_file(args.map).resolve(df) if args.map else None
    report = inspect_log(df, mapping)
    print(format_report(report))
    if args.save:
        report.to_frame().to_csv(args.save, index=False)
        print(f"\n-> {args.save}")
    return 0 if not any("必須" in w for w in report.warnings) else 1


def cmd_demo_log(args: argparse.Namespace) -> int:
    from .ingest import make_demo_log, make_j1939_demo_log

    if args.j1939:
        log = make_j1939_demo_log(sample_hz=args.sample_hz, seed=args.seed,
                                  naming=args.naming)
    else:
        log = make_demo_log(sample_hz=args.sample_hz, seed=args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    log.to_csv(out, index=False)
    time_col = "Timestamp" if "Timestamp" in log.columns else "time"
    print(f"{len(log)} samples ({log[time_col].iloc[-1]:.1f} s) -> {out}")
    return 0


def cmd_ollama(args: argparse.Namespace) -> int:
    from .config import PipelineConfig
    from .reporting import OllamaClient

    cfg = PipelineConfig.load(args.config)
    client = OllamaClient(host=cfg.ollama.host, model=cfg.ollama.model)
    ok = client.is_available()
    print(f"host      : {client.host}")
    print(f"available : {ok}")
    if ok:
        print("models    : " + ", ".join(client.available_models() or ["(none)"]))
        print(f"configured: {cfg.ollama.model}")
    else:
        print("ヒント: `ollama serve` を起動し、`ollama pull "
              f"{cfg.ollama.model}` でモデルを取得してください。")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="amtlab",
        description="AMT 変速品質の解析・適合最適化ツール "
                    "(scikit-learn / Optuna / seaborn / Ollama)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="解析パイプライン一式を実行")
    _add_common(p_run)
    p_run.add_argument("--samples", type=int, default=None, help="DoE のイベント数")
    p_run.add_argument("--model-trials", type=int, default=None, help="代理モデル探索の試行数")
    p_run.add_argument("--calib-trials", type=int, default=None, help="適合探索の試行数")
    p_run.add_argument("--mode", choices=["scalar", "pareto"], default=None)
    p_run.add_argument("--no-ollama", action="store_true", help="LLM レポートを使わない")
    p_run.add_argument("--quick", action="store_true", help="動作確認用の小規模実行")
    p_run.set_defaults(func=cmd_run)

    p_sim = sub.add_parser("simulate", help="単一変速のシミュレーションと時系列描画")
    p_sim.add_argument("--scenario", default="2-3@45,0.6",
                       help="from-to@speed_kmh,throttle[,grade_pct,payload_kg]")
    p_sim.add_argument("--control", default=None, help="適合値の YAML/JSON")
    p_sim.add_argument("--out", default="outputs/figures")
    p_sim.add_argument("--name", default="shift_trace")
    p_sim.add_argument("--trace-csv", action="store_true", help="時系列 CSV も保存")
    p_sim.set_defaults(func=cmd_simulate)

    p_ds = sub.add_parser("dataset", help="ランダム DoE データセットの生成")
    p_ds.add_argument("--samples", type=int, default=600)
    p_ds.add_argument("--seed", type=int, default=0)
    p_ds.add_argument("--jobs", type=int, default=-1)
    p_ds.add_argument("--out", default="outputs/data/dataset.csv")
    p_ds.set_defaults(func=cmd_dataset)

    p_an = sub.add_parser("analyze", help="既存データセットから解析パイプラインを実行")
    _add_common(p_an)
    p_an.add_argument("--dataset", required=True)
    p_an.add_argument("--no-ollama", action="store_true")
    p_an.set_defaults(func=cmd_analyze)

    p_cal = sub.add_parser("calibrate", help="適合値の最適化のみ実行")
    _add_common(p_cal)
    p_cal.add_argument("--trials", type=int, default=None)
    p_cal.add_argument("--mode", choices=["scalar", "pareto"], default=None)
    p_cal.set_defaults(func=cmd_calibrate)

    p_cmp = sub.add_parser("compare", help="グループ間(A社 vs B社 など)でログを比較")
    p_cmp.add_argument("--group", required=True, action="append", metavar="名前=パス",
                       help='比較するグループ (例: --group "A社=logs/a" '
                            '--group "B社=logs/b")。2 つ以上指定する')
    p_cmp.add_argument("--reference", default=None, help="基準グループ名(既定は先頭)")
    p_cmp.add_argument("--out", default="outputs")
    p_cmp.add_argument("--config", default=None)
    p_cmp.add_argument("--jobs", type=int, default=-1)
    p_cmp.add_argument("--boot", type=int, default=1000,
                       help="ファイル単位ブートストラップの反復回数")
    p_cmp.add_argument("--seed", type=int, default=0)
    p_cmp.add_argument("--resample-hz", type=float, default=100.0)
    p_cmp.add_argument("--map", default=None,
                       help="{内部名: 列名} の YAML/JSON で自動検出を上書き")
    p_cmp.add_argument("--no-auto", action="store_true")
    p_cmp.add_argument("--no-ollama", action="store_true")
    for name, default in (("col_time", "time"), ("col_engine_speed", "engine_speed_rpm"),
                          ("col_speed", "speed_kmh"), ("col_gear", "gear"),
                          ("col_throttle", "throttle"), ("col_torque", "shaft_torque")):
        p_cmp.add_argument(f"--{name.replace('_', '-')}", default=default)
    p_cmp.set_defaults(func=cmd_compare)

    p_rep = sub.add_parser("report", help="summary.json からレポートを再生成")
    p_rep.add_argument("--config", default=None)
    p_rep.add_argument("--summary", default="outputs/summary.json")
    p_rep.add_argument("--out", default="outputs/report.md")
    p_rep.add_argument("--model", default=None, help="Ollama モデル名を上書き")
    p_rep.add_argument("--no-ollama", action="store_true")
    p_rep.set_defaults(func=cmd_report)

    p_in = sub.add_parser("ingest", help="実車ログから変速イベントを切り出して解析")
    p_in.add_argument("--log", required=True, nargs="+",
                      help="ログのパス / ディレクトリ / glob(複数可、parquet 対応)")
    p_in.add_argument("--jobs", type=int, default=-1, help="並列数(複数ファイル時)")
    p_in.add_argument("--out", default="outputs")
    p_in.add_argument("--resample-hz", type=float, default=100.0)
    p_in.add_argument("--col-time", default="time")
    p_in.add_argument("--col-engine-speed", default="engine_speed_rpm")
    p_in.add_argument("--col-speed", default="speed_kmh")
    p_in.add_argument("--col-gear", default="gear")
    p_in.add_argument("--col-throttle", default="throttle")
    p_in.add_argument("--col-torque", default="shaft_torque")
    p_in.add_argument("--map", default=None,
                      help="{内部名: 列名} の YAML/JSON で自動検出を上書き")
    p_in.add_argument("--no-auto", action="store_true",
                      help="J1939 の列名自動検出を無効にする")
    p_in.set_defaults(func=cmd_ingest)

    p_dl = sub.add_parser("demo-log", help="取り込み確認用のデモログ CSV を生成")
    p_dl.add_argument("--out", default="outputs/data/demo_log.csv")
    p_dl.add_argument("--sample-hz", type=float, default=100.0)
    p_dl.add_argument("--seed", type=int, default=0)
    p_dl.add_argument("--j1939", action="store_true",
                      help="J1939 の信号名・単位・更新周期で出力する")
    p_dl.add_argument("--naming", choices=["short", "long"], default="short",
                      help="信号名の書き方 (short: TransShiftInProcess)")
    p_dl.set_defaults(func=cmd_demo_log)

    p_ins = sub.add_parser("inspect", help="ログの信号構成とサンプルレートを診断")
    p_ins.add_argument("--log", required=True, nargs="+",
                       help="ログのパス / ディレクトリ / glob(複数可)")
    p_ins.add_argument("--save", default=None,
                       help="チャネル診断の CSV 出力先(複数ファイル時は "
                            "<stem>_presence.csv に信号の有無表も出力)")
    p_ins.add_argument("--map", default=None,
                       help="{内部名: 列名} の YAML/JSON で自動検出を上書き")
    p_ins.set_defaults(func=cmd_inspect)

    p_ol = sub.add_parser("ollama", help="Ollama の接続確認")
    p_ol.add_argument("--config", default=None)
    p_ol.set_defaults(func=cmd_ollama)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
