import numpy as np
import pytest

from retrieval.statistics import (
    classify_differences,
    cluster_bootstrap_ci,
    cluster_sign_flip_test,
    holm_adjust,
    positive_rank,
    reconstruct_aggregate,
    validate_clusters,
    validate_query_alignment,
)


def test_cluster_alignment_rejects_duplicates_and_missing_queries():
    validate_clusters(["a", "b", "c"], [["a", "b"], ["c"]])
    with pytest.raises(ValueError):
        validate_clusters(["a", "b"], [["a"], ["a", "b"]])
    with pytest.raises(ValueError):
        validate_clusters(["a", "b"], [["a"]])


def test_cluster_bootstrap_is_reproducible_and_cluster_aware():
    values = [1.0, 1.0, -1.0]
    query_ids = ["a", "b", "c"]
    clusters = [["a", "b"], ["c"]]
    first = cluster_bootstrap_ci(values, query_ids, clusters, replicates=1000, seed=7)
    second = cluster_bootstrap_ci(values, query_ids, clusters, replicates=1000, seed=7)
    assert first == second
    assert first["lower"] <= np.mean(values) <= first["upper"]


def test_cluster_sign_flip_uses_plus_one_correction():
    result = cluster_sign_flip_test(
        [1.0, 1.0], ["a", "b"], [["a", "b"]], permutations=10, seed=3,
    )
    assert result["p_value"] == (result["extreme_count"] + 1) / 11
    assert result["p_value"] > 0


def test_tolerance_classification():
    assert classify_differences([2e-12, 1e-13, -1e-13, -2e-12]) == {
        "improved": 1, "unchanged": 2, "degraded": 1,
    }


def test_multiple_positive_rank_and_missing_marker():
    ranking = ["d2", "d1", "d3"]
    assert [positive_rank(ranking, doc) for doc in ["d1", "d2", "missing"]] == [2, 1, ">100"]


def test_holm_adjustment_is_monotone_and_bounded():
    adjusted = holm_adjust({"a": 0.001, "b": 0.02, "c": 0.5})
    assert adjusted["a"] == pytest.approx(0.003)
    assert adjusted["b"] == pytest.approx(0.04)
    assert adjusted["c"] == pytest.approx(0.5)


def test_exact_query_alignment():
    validate_query_alignment(["a", "b"], {"zero": {"a": {}, "b": {}}, "trained": {"b": {}, "a": {}}})
    with pytest.raises(ValueError):
        validate_query_alignment(["a", "b"], {"missing": {"a": {}}})
    with pytest.raises(ValueError):
        validate_query_alignment(["a", "b"], {"extra": {"a": {}, "b": {}, "c": {}}})


def test_aggregate_reconstruction():
    per_query = {"a": {"NDCG@10": 0.25}, "b": {"NDCG@10": 0.75}}
    assert reconstruct_aggregate(per_query, "NDCG@10") == pytest.approx(0.5)
