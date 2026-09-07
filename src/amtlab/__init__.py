"""amtlab: AMT(自動MT)車両の変速品質データ解析・適合最適化ライブラリ。

主要モジュール
--------------
``amtlab.simulation``  車両/変速制御のプラントモデル
``amtlab.dataset``     実験計画(DoE)によるデータ生成
``amtlab.features``    変速品質 KPI の抽出
``amtlab.modeling``    scikit-learn 代理モデル
``amtlab.tuning``      Optuna によるハイパーパラメータ探索
``amtlab.calibration`` Optuna による適合値の多目的最適化
``amtlab.viz``         seaborn 可視化
``amtlab.reporting``   Ollama によるレポート生成
``amtlab.pipeline``    一連の解析パイプライン
"""

from .config import PipelineConfig
from .features import extract_kpis, results_to_frame
from .simulation import ShiftControlParams, ShiftScenario, simulate_shift

__version__ = "0.1.0"

__all__ = [
    "PipelineConfig",
    "ShiftControlParams",
    "ShiftScenario",
    "extract_kpis",
    "results_to_frame",
    "simulate_shift",
    "__version__",
]
