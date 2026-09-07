"""解析結果のサマリ作成と、ローカル Ollama によるレポート生成。

本プロジェクトは「手元の Ollama を前提にした解析基盤」を想定している。
外部 API に計測データを出さずに、解析結果の考察文だけをローカル LLM に
書かせる。Ollama が無い環境でも決定論的なテンプレートレポートを出力する
ため、パイプラインが止まることはない。
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

SYSTEM_PROMPT = (
    "あなたは自動車のパワートレイン適合(キャリブレーション)エンジニアです。"
    "AMT(自動MT)の変速品質データ解析の結果を、事実に基づいて簡潔に考察します。"
    "与えられた数値以外を推測で作らないでください。"
)

REPORT_INSTRUCTION = """以下は AMT 変速品質の解析結果(JSON)です。
Markdown で技術レポートを書いてください。構成は次の通りです。

1. サマリ(3行以内)
2. 変速品質を支配している因子(permutation importance と相関から)
3. カレントギア(現在ギア)別の傾向 — どの段の変速時間・車速落ち込みが問題か
4. 最適化された適合値の妥当性(ベースライン比の改善と悪化のトレードオフ)
5. 残課題と次に取るべき計測・実験

車速の指標は 2 種類ある。speed_drop_kmh は実測波形のピーク→谷の落ち込み幅、
speed_loss_kmh は「変速しなかった場合」との差(伸びの停滞を含む)。混同しないこと。

数値は JSON の値をそのまま引用し、単位を明記してください。
"""


@dataclass
class OllamaClient:
    """依存を増やさないための最小 Ollama クライアント(標準ライブラリのみ)。"""

    host: str = "http://localhost:11434"
    model: str = "qwen2.5:7b"
    temperature: float = 0.3
    timeout_s: float = 180.0

    def __post_init__(self) -> None:
        self.host = os.environ.get("OLLAMA_HOST", self.host).rstrip("/")
        if not self.host.startswith("http"):
            self.host = f"http://{self.host}"

    def _request(self, path: str, payload: dict[str, Any] | None = None,
                 timeout: float | None = None) -> dict[str, Any]:
        url = f"{self.host}{path}"
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"},
            method="POST" if data else "GET",
        )
        with urllib.request.urlopen(req, timeout=timeout or self.timeout_s) as res:
            return json.loads(res.read().decode("utf-8"))

    def is_available(self) -> bool:
        try:
            self._request("/api/tags", timeout=3.0)
            return True
        except (urllib.error.URLError, OSError, ValueError, TimeoutError):
            return False

    def available_models(self) -> list[str]:
        try:
            tags = self._request("/api/tags", timeout=5.0)
        except Exception:
            return []
        return [m.get("name", "") for m in tags.get("models", [])]

    def chat(self, prompt: str, system: str = SYSTEM_PROMPT) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        res = self._request("/api/chat", payload)
        return res.get("message", {}).get("content", "").strip()


# ----------------------------------------------------------------------
def build_summary(
    dataset: pd.DataFrame,
    surrogates: dict[str, Any],
    importance: pd.DataFrame,
    calibration_outcome,
    tuning: dict[str, Any] | None = None,
    gears: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """LLM とテンプレートの双方に渡す解析サマリ(JSON 化可能)を作る。"""

    kpis = [c for c in ("shift_time_s", "speed_drop_kmh", "speed_loss_kmh",
                        "torque_interrupt_s", "jerk_rms", "jerk_peak")
            if c in dataset.columns]
    describe = dataset[kpis].describe().loc[["mean", "std", "min", "max"]].round(3)

    imp = importance.head(8)[["feature", "importance", "kind"]].round(4)
    improvement = calibration_outcome.improvement().round(3)

    summary: dict[str, Any] = {
        "dataset": {
            "n_events": int(len(dataset)),
            "upshift_ratio": round(float(dataset["is_upshift"].mean()), 3),
            "speed_range_kmh": [round(float(dataset["speed_kmh"].min()), 1),
                                round(float(dataset["speed_kmh"].max()), 1)],
            "kpi_statistics": json.loads(describe.to_json()),
        },
        "surrogate_models": {
            name: {k: round(float(v), 4) for k, v in model.metrics.items()}
            | {"model": model.model_name}
            for name, model in surrogates.items()
        },
        "top_importance": imp.to_dict(orient="records"),
        "calibration": {
            "mode": calibration_outcome.mode,
            "n_trials": len(calibration_outcome.study.trials),
            "baseline_params": calibration_outcome.baseline_control.as_dict(),
            "optimized_params": {
                k: round(v, 3) for k, v in calibration_outcome.best_control.as_dict().items()
            },
            "improvement": improvement.to_dict(orient="records"),
            "n_pareto_solutions": int(len(calibration_outcome.pareto)),
        },
    }
    if gears is not None and len(gears):
        cols = [c for c in gears.columns if c.endswith("_mean") or c in
                ("current_gear", "direction")]
        summary["by_current_gear"] = gears[cols].round(3).to_dict(orient="records")
    if tuning:
        summary["hyperparameter_tuning"] = tuning
    return summary


def fallback_report(summary: dict[str, Any]) -> str:
    """Ollama が使えない場合の決定論的レポート。"""

    cal = summary["calibration"]
    lines = [
        "# AMT 変速品質 解析レポート",
        "",
        "> Ollama に接続できなかったため、テンプレートによる自動生成レポートです。",
        "",
        "## 1. サマリ",
        f"- 解析対象: ランダム DoE による変速イベント {summary['dataset']['n_events']} 件"
        f"(車速 {summary['dataset']['speed_range_kmh'][0]}〜"
        f"{summary['dataset']['speed_range_kmh'][1]} km/h)",
        "- 代理モデル: "
        + ", ".join(
            f"{k} → {v['model']} (R²={v['r2']:.3f}, RMSE={v['rmse']:.3f})"
            for k, v in summary["surrogate_models"].items()
        ),
        f"- 適合最適化: {cal['mode']} モード / {cal['n_trials']} 試行",
        "",
        "## 2. 変速品質の支配因子 (permutation importance 上位)",
        "",
        "| 因子 | 種別 | 重要度 (R² 低下) |",
        "| --- | --- | --- |",
    ]
    for row in summary["top_importance"]:
        lines.append(f"| {row['feature']} | {row['kind']} | {row['importance']:.4f} |")

    if summary.get("by_current_gear"):
        lines += ["", "## 3. カレントギア(現在ギア)別の傾向", "",
                  "| 現在ギア | 方向 | 変速時間 [s] | 車速落ち込み [km/h] | 車速損失 [km/h] |",
                  "| --- | --- | --- | --- | --- |"]
        for row in summary["by_current_gear"]:
            lines.append(
                f"| {row['current_gear']} | {row['direction']} | "
                f"{row.get('shift_time_s_mean', float('nan')):.2f} | "
                f"{row.get('speed_drop_kmh_mean', float('nan')):.2f} | "
                f"{row.get('speed_loss_kmh_mean', float('nan')):.2f} |"
            )
        lines += ["",
                  "※ 車速落ち込み = 実測のピーク→谷、車速損失 = 変速しなかった場合との差。"]

    lines += ["", "## 4. 適合値の最適化結果", "",
              "| 指標 | ベースライン | 最適化後 | 改善率 [%] |", "| --- | --- | --- | --- |"]
    for row in cal["improvement"]:
        lines.append(
            f"| {row['metric']} | {row['baseline']:.3f} | {row['optimized']:.3f} | "
            f"{row['improvement_pct']:.1f} |"
        )

    lines += ["", "### 最適化された適合値", "", "| パラメータ | ベースライン | 最適化後 |",
              "| --- | --- | --- |"]
    for key, value in cal["optimized_params"].items():
        lines.append(f"| {key} | {cal['baseline_params'][key]:.3f} | {value:.3f} |")

    lines += [
        "",
        "## 5. 残課題",
        "- 実車ログでのモデル妥当性確認(プラントモデルのパラメータ同定)",
        "- 低速段(1→2速)大トルク域のショック低減",
        "- クラッチ発熱・耐久を含めた長時間走行での検証",
        "",
    ]
    return "\n".join(lines)


def generate_report(
    summary: dict[str, Any],
    client: OllamaClient | None = None,
    enabled: bool = True,
    language: str = "ja",
) -> tuple[str, str]:
    """レポート本文と生成元 ("ollama" / "fallback") を返す。"""

    if not enabled:
        return fallback_report(summary), "fallback"
    client = client or OllamaClient()
    if not client.is_available():
        return fallback_report(summary), "fallback"

    lang_note = "" if language == "ja" else f"\n出力言語: {language}\n"
    prompt = (
        REPORT_INSTRUCTION
        + lang_note
        + "\n```json\n"
        + json.dumps(summary, ensure_ascii=False, indent=2)
        + "\n```\n"
    )
    try:
        text = client.chat(prompt)
    except Exception as exc:  # pragma: no cover - ネットワーク依存
        return (
            fallback_report(summary)
            + f"\n\n> Ollama 生成に失敗しました: {type(exc).__name__}: {exc}\n",
            "fallback",
        )
    if not text:
        return fallback_report(summary), "fallback"
    return text, "ollama"


def write_report(text: str, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
