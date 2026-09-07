from amtlab.tuning import study_to_frame, tune_surrogate


def test_tuning_finds_a_model(small_dataset):
    outcome = tune_surrogate(small_dataset, target="shift_time_s", n_trials=5,
                             model_names=["ridge", "random_forest"], n_splits=3)
    assert outcome.best_model_name in ("ridge", "random_forest")
    assert outcome.best_rmse > 0
    assert outcome.surrogate.target == "shift_time_s"


def test_history_frame_tracks_best_so_far(small_dataset):
    outcome = tune_surrogate(small_dataset, target="jerk_rms", n_trials=4,
                             model_names=["ridge"], n_splits=3)
    hist = study_to_frame(outcome.study)
    assert len(hist) == 4
    assert list(hist["best_so_far"]) == sorted(hist["best_so_far"], reverse=True)
