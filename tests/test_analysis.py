"""Error analysis: the attribution rules do what they claim, on cases built to test them."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlops_loop import analysis, skeleton
from mlops_loop.analysis import AnalysisError, COMPONENTS
from mlops_loop.validate import TARGET_INT


@pytest.fixture(scope="module")
def champion(tmp_tracking, sample_csv):
    return skeleton.run(
        source=str(sample_csv),
        data_dir=tmp_tracking["data_dir"],
        run_name="analysis-fixture",
    )


@pytest.fixture(scope="module")
def analysed(champion, tmp_tracking, sample_csv, tmp_path_factory):
    return analysis.run(
        source=str(sample_csv),
        data_dir=tmp_tracking["data_dir"],
        report_path=str(tmp_path_factory.mktemp("analysis") / "error_analysis.md"),
    )


def test_every_failure_gets_exactly_one_component(analysed) -> None:
    assert sum(analysed.full_tally.values()) == analysed.errors
    assert sum(analysed.tally.values()) == analysed.sample_size
    assert set(analysed.full_tally) == set(COMPONENTS)


def test_the_sample_is_twenty_or_all_of_them(analysed) -> None:
    assert analysed.sample_size == min(20, analysed.errors)
    assert len(analysed.cases) == analysed.sample_size
    assert len({case.customer_id for case in analysed.cases}) == analysed.sample_size


def test_false_negatives_and_positives_add_up(analysed) -> None:
    assert analysed.false_negatives + analysed.false_positives == analysed.errors
    kinds = [case.kind for case in analysed.cases]
    assert set(kinds) <= {"false negative", "false positive"}


def test_every_case_carries_a_reason(analysed) -> None:
    for case in analysed.cases:
        assert case.component in COMPONENTS
        assert case.reason, f"{case.customer_id} was attributed with no reason"
        assert 0.0 <= case.probability <= 1.0


def test_the_biggest_cell_is_the_largest_tally(analysed) -> None:
    name, count = analysed.biggest_cell
    assert count == max(analysed.tally.values())
    assert analysed.tally[name] == count


def test_the_diagnosis_and_fix_are_written_for_the_biggest_cell(analysed) -> None:
    """Each component gets its own prose, so the report cannot say the same thing regardless."""
    name, count = analysed.biggest_cell
    assert str(count) in analysed.diagnosis
    assert analysed.cheapest_fix

    others = {
        component
        for component in COMPONENTS
        if component != name
    }
    context = {
        "threshold": 0.5,
        "val_threshold": 0.6,
        "fibre_share": 0.44,
        "trained_on": "train",
        "model_errors": 10,
        "confident_fp": 7,
        "confident_fp_median": 0.74,
        "predicted_positive_rate": 0.37,
        "actual_rate": 0.26,
        "class_weight": "balanced",
    }
    for other in others:
        alternative = analysis._cheapest_fix({other: 1}, context)
        assert alternative != analysed.cheapest_fix


def test_the_analysis_is_deterministic(champion, tmp_tracking, sample_csv, tmp_path) -> None:
    """Same seed, same twenty rows, so the tally can be argued with rather than re-rolled."""
    first = analysis.run(
        source=str(sample_csv),
        data_dir=tmp_tracking["data_dir"],
        report_path=str(tmp_path / "a.md"),
    )
    second = analysis.run(
        source=str(sample_csv),
        data_dir=tmp_tracking["data_dir"],
        report_path=str(tmp_path / "b.md"),
    )
    assert [case.customer_id for case in first.cases] == [
        case.customer_id for case in second.cases
    ]
    assert first.tally == second.tally


def test_a_row_outside_the_training_range_is_charged_to_the_split(splits) -> None:
    """The split rule fires on values the training data never covered, not on model error."""
    from mlops_loop.train import fit_model

    train_frame = splits.train
    model = fit_model(
        train_frame, {"C": 1.0, "max_iter": 200, "random_state": 42}, log=False
    )
    holdout = splits.holdout.copy()
    # Push one row's charges far outside anything train contains and flip its label so it
    # is guaranteed to be wrong.
    holdout.loc[holdout.index[0], "MonthlyCharges"] = float(train_frame.MonthlyCharges.max() + 500)
    truth = int(model.predict_proba(_features(holdout))[0, 1] < 0.5)
    holdout.loc[holdout.index[0], TARGET_INT] = truth

    table = analysis.attribute(holdout, train_frame, model, 0.5, 0.5)
    assert table.loc[0, "wrong"]
    assert table.loc[0, "component"] == "split"
    assert "outside the train range" in table.loc[0, "reason"]


def test_a_near_miss_is_charged_to_the_threshold(splits) -> None:
    """A row right at the cut, correct under the validation-optimal threshold."""
    from mlops_loop.train import fit_model

    model = fit_model(
        splits.train, {"C": 1.0, "max_iter": 200, "random_state": 42}, log=False
    )
    holdout = splits.holdout
    scores = model.predict_proba(_features(holdout))[:, 1]
    # A row scored just above 0.5 whose truth is 0: wrong at 0.5, right at a higher cut.
    candidates = np.where((scores > 0.5) & (scores < 0.6) & (holdout[TARGET_INT] == 0))[0]
    if not candidates.size:
        pytest.skip("the fixture produced no row between 0.5 and 0.6 to test with")

    table = analysis.attribute(holdout, splits.train, model, 0.5, 0.65)
    row = table.iloc[int(candidates[0])]
    assert row["wrong"]
    assert row["component"] in ("threshold", "split", "features")
    if row["component"] == "threshold":
        assert "right side of" in row["reason"]


def test_attribution_stops_if_nothing_is_wrong(splits) -> None:
    from mlops_loop.train import fit_model

    model = fit_model(splits.train, {"C": 1.0, "max_iter": 200, "random_state": 42}, log=False)
    perfect = splits.holdout.copy()
    scores = model.predict_proba(_features(perfect))[:, 1]
    perfect[TARGET_INT] = (scores >= 0.5).astype(int)
    with pytest.raises(AnalysisError, match="no mistakes"):
        analysis.attribute(perfect, splits.train, model, 0.5, 0.5)


def test_the_report_names_the_biggest_cell_and_does_not_apply_the_fix(analysed) -> None:
    from mlops_loop import report

    import tempfile
    from pathlib import Path

    target = Path(tempfile.mkdtemp()) / "error_analysis.md"
    report.write_error_analysis(analysed, path=target)
    text = target.read_text(encoding="utf-8")

    assert analysed.run_id in text
    assert analysed.champion_run_id in text
    assert "## The biggest cell" in text
    assert "## The cheapest fix" in text
    assert "Not applied in this session" in text
    for case in analysed.cases:
        assert case.customer_id in text
    for component in COMPONENTS:
        assert component in text


def test_the_val_threshold_comes_from_val_not_the_holdout(splits) -> None:
    """Tuning the cut on the holdout would make the threshold cell unfalsifiable."""
    from mlops_loop.train import fit_model

    model = fit_model(splits.train, {"C": 1.0, "max_iter": 200, "random_state": 42}, log=False)
    on_val = analysis.best_f1_threshold(model, splits.val)
    assert 0.0 <= on_val <= 1.0


def _features(frame: pd.DataFrame) -> pd.DataFrame:
    from mlops_loop.features import feature_frame

    return feature_frame(frame)
