import pytest

from amtlab.dataset import CONDITION_FEATURES, FEATURE_COLUMNS
from amtlab.modeling import (
    SurrogateModel,
    build_pipeline,
    condition_only_baseline,
    cross_validate_model,
    fit_surrogate,
    importance_frame,
    partial_dependence_frame,
)


@pytest.fixture(scope="module")
def surrogate(small_dataset):
    return fit_surrogate(small_dataset, target="shift_time_s", model_name="random_forest",
                         n_splits=3, n_estimators=60)


def test_unknown_model_name_raises():
    with pytest.raises(KeyError):
        build_pipeline("magic_regressor")


def test_surrogate_learns_something(surrogate):
    assert surrogate.metrics["r2"] > 0.3
    assert surrogate.metrics["rmse"] < surrogate.metrics["target_std"]
    assert len(surrogate.cv_predictions) > 0


def test_prediction_shape(surrogate, small_dataset):
    preds = surrogate.predict(small_dataset)
    assert preds.shape == (len(small_dataset),)


def test_cross_validation_returns_oof_predictions(small_dataset):
    metrics, preds = cross_validate_model(small_dataset, "jerk_rms", "ridge", n_splits=3)
    assert set(("rmse", "mae", "r2", "target_std")) <= set(metrics)
    assert len(preds) == len(small_dataset)
    assert "residual" in preds.columns


def test_importance_is_sorted_and_labelled(surrogate, small_dataset):
    imp = importance_frame(surrogate, small_dataset, n_repeats=3)
    assert list(imp["importance"]) == sorted(imp["importance"], reverse=True)
    assert set(imp["kind"]) <= {"calibration", "condition"}
    assert len(imp) == len(FEATURE_COLUMNS)


def test_partial_dependence_frame(surrogate, small_dataset):
    pdp = partial_dependence_frame(surrogate, small_dataset,
                                   features=["clutch_close_rate"], grid_resolution=5)
    assert set(pdp["feature"]) == {"clutch_close_rate"}
    assert len(pdp) == 5


def test_condition_only_baseline_uses_fewer_features(small_dataset):
    metrics = condition_only_baseline(small_dataset, "shift_time_s", n_splits=3)
    assert "r2" in metrics
    assert len(CONDITION_FEATURES) < len(FEATURE_COLUMNS)


def test_surrogate_roundtrip(surrogate, tmp_path, small_dataset):
    path = surrogate.save(tmp_path / "m.joblib")
    loaded = SurrogateModel.load(path)
    assert loaded.target == surrogate.target
    assert loaded.predict(small_dataset.head(3)).shape == (3,)
