import math
import pytest

from src.audit import downstream_task_leakage as dtl
from src.audit.downstream_task_leakage import (
    _aggregate_patient_groups,
    _build_audit_mlp_arrays,
    _build_label_stats,
    _extract_record_subgroups,
    _is_best,
    _multilabel_utility_scores,
    _patient_mia_from_losses,
    _rewrite_record_for_feature_source,
    _sanitize_config_tree,
)


def test_patient_mia_from_losses_uses_members_as_positive_class():
    out = _patient_mia_from_losses(
        member_losses=[0.10, 0.12, 0.15],
        member_ids=["m1", "m2", "m3"],
        nonmember_losses=[0.80, 0.90, 1.00],
        nonmember_ids=["n1", "n2", "n3"],
        agg="mean",
        fpr_targets=[0.01],
        bootstrap_iters=200,
        bootstrap_confidence=0.95,
        bootstrap_seed=17,
    )
    assert out["auc"] > 0.99
    assert out["auc_ci"]["lower"] > 0.95
    assert len(out["tpr_at_fpr"]) == 1
    assert math.isclose(out["tpr_at_fpr"][0]["tpr"], 1.0, rel_tol=1e-9)
    assert out["tpr_at_fpr"][0]["empirical_fpr"] <= 1.0


def test_rewrite_record_for_synthetic_only_zeros_raw_channels():
    record = {
        "conditions": [["C1"], ["C2", "C3"]],
        "procedures": [["P1"], ["P2"]],
        "drugs_hist": [["D1"], ["D2"]],
        "new_conditions": [["SC1"], ["SC2"]],
        "new_procedures": [["SP1"], ["SP2"]],
        "new_drugs": [["SD1"], ["SD2"]],
        "labels": ["L1"],
    }
    out = _rewrite_record_for_feature_source(record, "synthetic_only")
    assert out["conditions"] == [["SC1"], ["SC2"]]
    assert out["procedures"] == [["SP1"], ["SP2"]]
    assert out["drugs_hist"] == [["SD1"], ["SD2"]]
    assert out["new_conditions"] == [[], []]
    assert out["new_procedures"] == [[], []]
    assert out["new_drugs"] == [[], []]
    assert out["labels"] == ["L1"]


def test_rewrite_record_for_raw_only_zeros_new_channels():
    record = {
        "conditions": [["C1"], ["C2", "C3"]],
        "procedures": [["P1"], ["P2"]],
        "drugs_hist": [["D1"], ["D2"]],
        "new_conditions": [["SC1"], ["SC2"]],
        "new_procedures": [["SP1"], ["SP2"]],
        "new_drugs": [["SD1"], ["SD2"]],
        "labels": ["L1", "L2"],
    }
    out = _rewrite_record_for_feature_source(record, "raw_only")
    assert out["conditions"] == [["C1"], ["C2", "C3"]]
    assert out["procedures"] == [["P1"], ["P2"]]
    assert out["drugs_hist"] == [["D1"], ["D2"]]
    assert out["new_conditions"] == [[], []]
    assert out["new_procedures"] == [[], []]
    assert out["new_drugs"] == [[], []]
    assert out["labels"] == ["L1", "L2"]


def test_subgroup_bins_and_patient_aggregation_are_ordered():
    records = [
        {
            "patient_id": "p1",
            "visit_id_hist": ["v1"],
            "labels": ["A"],
        },
        {
            "patient_id": "p1",
            "visit_id_hist": ["v1", "v2", "v3", "v4"],
            "labels": ["A", "B", "C", "D"],
        },
        {
            "patient_id": "p2",
            "visit_id_hist": ["v1", "v2"],
            "labels": ["A", "B"],
        },
    ]
    stats = _build_label_stats(records)
    enriched = [{**record, "subgroups": _extract_record_subgroups(record, stats)} for record in records]
    groups = _aggregate_patient_groups(enriched)

    assert groups["history_len_bin"]["p1"] == "03:hist_4to7"
    assert groups["label_size_bin"]["p1"] == "03:label_4to5"
    assert groups["label_pattern_bin"]["p1"] == "03:pattern_singleton"
    assert groups["history_len_bin"]["p2"] == "02:hist_2to3"


def test_is_best_supports_max_and_min():
    assert _is_best(0.2, 0.3, "max") is True
    assert _is_best(0.2, 0.1, "max") is False
    assert _is_best(0.2, 0.1, "min") is True
    assert _is_best(0.2, 0.3, "min") is False


def test_sanitize_config_tree_drops_callables_and_classmethods():
    class Demo:
        @classmethod
        def cmethod(cls):
            return 1

    cfg = {
        "TASK": "PHE",
        "nested": {"ok": 1, "call": lambda: None},
        "get_params": Demo.__dict__["cmethod"],
    }
    out = _sanitize_config_tree(cfg)
    assert out == {"TASK": "PHE", "nested": {"ok": 1}}


def test_audit_mlp_helpers_vectorize_records_and_score_multilabel_predictions():
    if dtl.np is None:
        pytest.skip("numpy not available in lightweight local test environment")
    train_records = [
        {
            "patient_id": "p1",
            "conditions": [["C1"], ["C2"]],
            "procedures": [["P1"]],
            "drugs_hist": [["D1"]],
            "labels": ["L1", "L2"],
        },
        {
            "patient_id": "p2",
            "conditions": [["C3"]],
            "procedures": [[]],
            "drugs_hist": [["D2"]],
            "labels": ["L2"],
        },
    ]
    test_records = [
        {
            "patient_id": "p3",
            "conditions": [["C1"]],
            "procedures": [["P2"]],
            "drugs_hist": [["D3"]],
            "labels": ["L1"],
        }
    ]
    enc = _build_audit_mlp_arrays(train_records, test_records, ["conditions", "procedures", "drugs_hist"])
    assert enc["x_train"].shape == (2, 8)
    assert enc["y_train"].shape == (2, 2)
    scores = _multilabel_utility_scores(enc["y_train"], enc["y_train"])
    assert math.isclose(scores["jaccard_samples"], 1.0, rel_tol=1e-9)
    assert math.isclose(scores["f1_samples"], 1.0, rel_tol=1e-9)
