import warnings

import pytest

from amtlab.dataset import build_dataset

warnings.filterwarnings("ignore", category=FutureWarning)


@pytest.fixture(scope="session")
def small_dataset():
    """テスト全体で使い回す小規模 DoE データセット。"""
    return build_dataset(n_samples=150, seed=7, n_jobs=-1)
