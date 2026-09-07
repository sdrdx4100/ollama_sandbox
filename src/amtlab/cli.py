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
        weights=cfg.calibration.weights, worst_case_weight=cfg.calibration.worst_case_weight
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
        print(f"figure: {viz.plot_pareto(outcome.pareto, out / 'figures')}")
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
    from .ingest import SignalMap, analyze_log

    sm = SignalMap(
        time=args.col_time,
        engine_speed_rpm=args.col_engine_speed,
        speed_kmh=args.col_speed,
        gear=args.col_gear,
        throttle=args.col_throttle,
        shaft_torque=args.col_torque,
    )
    trace, kpi = analyze_log(args.log, sm, resample_hz=args.resample_hz)
    out = Path(args.out)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    kpi.to_csv(out / "tables" / "log_events.csv", index=False)
    if kpi.empty:
        print("変速イベントを検出できませんでした(ギヤ信号を確認してください)")
        return 1
    print(kpi.round(3).to_string(index=False))
    print(f"figure: {viz.plot_log_overview(trace, kpi, out / 'figures')}")
    print(f"figure: {viz.plot_kpi_distributions(kpi, out / 'figures', 'log_kpi_distributions')}")
    return 0


def cmd_demo_log(args: argparse.Namespace) -> int:
    from .ingest import make_demo_log

    log = make_demo_log(sample_hz=args.sample_hz, seed=args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    log.to_csv(out, index=False)
    print(f"{len(log)} samples ({log['time'].iloc[-1]:.1f} s) -> {out}")
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

    p_rep = sub.add_parser("report", help="summary.json からレポートを再生成")
    p_rep.add_argument("--config", default=None)
    p_rep.add_argument("--summary", default="outputs/summary.json")
    p_rep.add_argument("--out", default="outputs/report.md")
    p_rep.add_argument("--model", default=None, help="Ollama モデル名を上書き")
    p_rep.add_argument("--no-ollama", action="store_true")
    p_rep.set_defaults(func=cmd_report)

    p_in = sub.add_parser("ingest", help="実車ログから変速イベントを切り出して解析")
    p_in.add_argument("--log", required=True, help="ログ CSV のパス")
    p_in.add_argument("--out", default="outputs")
    p_in.add_argument("--resample-hz", type=float, default=100.0)
    p_in.add_argument("--col-time", default="time")
    p_in.add_argument("--col-engine-speed", default="engine_speed_rpm")
    p_in.add_argument("--col-speed", default="speed_kmh")
    p_in.add_argument("--col-gear", default="gear")
    p_in.add_argument("--col-throttle", default="throttle")
    p_in.add_argument("--col-torque", default="shaft_torque")
    p_in.set_defaults(func=cmd_ingest)

    p_dl = sub.add_parser("demo-log", help="取り込み確認用のデモログ CSV を生成")
    p_dl.add_argument("--out", default="outputs/data/demo_log.csv")
    p_dl.add_argument("--sample-hz", type=float, default=100.0)
    p_dl.add_argument("--seed", type=int, default=0)
    p_dl.set_defaults(func=cmd_demo_log)

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
