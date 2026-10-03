"""Cluster-aware paired statistics used by the frozen DEV analysis."""
from __future__ import annotations

import numpy as np


TOLERANCE = 1e-12


def validate_query_alignment(expected_query_ids, systems):
    expected = set(expected_query_ids)
    if len(expected) != len(expected_query_ids):
        raise ValueError("Expected query IDs contain duplicates")
    for name, values in systems.items():
        if set(values) != expected or len(values) != len(expected):
            raise ValueError(f"Query alignment differs for {name}")


def reconstruct_aggregate(per_query, metric):
    if not per_query:
        raise ValueError("Cannot reconstruct an aggregate from no queries")
    return float(np.mean([values[metric] for values in per_query.values()]))


def classify_differences(values, tolerance=TOLERANCE):
    values = np.asarray(values, dtype=float)
    return {
        "improved": int(np.sum(values > tolerance)),
        "unchanged": int(np.sum(np.abs(values) <= tolerance)),
        "degraded": int(np.sum(values < -tolerance)),
    }


def validate_clusters(query_ids, clusters):
    expected = set(query_ids)
    flattened = [query_id for cluster in clusters for query_id in cluster]
    if len(flattened) != len(set(flattened)):
        raise ValueError("A query appears in more than one cluster")
    if set(flattened) != expected:
        raise ValueError("Clusters do not exactly cover the evaluated queries")


def cluster_arrays(differences, query_ids, clusters):
    validate_clusters(query_ids, clusters)
    index = {query_id: position for position, query_id in enumerate(query_ids)}
    values = np.asarray(differences, dtype=float)
    sums = np.asarray([sum(values[index[q]] for q in cluster) for cluster in clusters])
    sizes = np.asarray([len(cluster) for cluster in clusters], dtype=int)
    return sums, sizes


def cluster_bootstrap_ci(
    differences, query_ids, clusters, *, replicates=10_000, seed=20261005,
):
    if replicates < 1:
        raise ValueError("replicates must be positive")
    sums, sizes = cluster_arrays(differences, query_ids, clusters)
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(clusters), size=(replicates, len(clusters)))
    means = sums[samples].sum(axis=1) / sizes[samples].sum(axis=1)
    return {
        "lower": float(np.percentile(means, 2.5)),
        "upper": float(np.percentile(means, 97.5)),
        "replicates": replicates,
        "seed": seed,
    }


def cluster_sign_flip_test(
    differences, query_ids, clusters, *, permutations=50_000, seed=20261006,
):
    if permutations < 1:
        raise ValueError("permutations must be positive")
    sums, _ = cluster_arrays(differences, query_ids, clusters)
    observed = abs(float(np.mean(differences)))
    rng = np.random.default_rng(seed)
    signs = rng.integers(0, 2, size=(permutations, len(clusters)), dtype=np.int8)
    signs = signs * 2 - 1
    null_statistics = np.abs((signs * sums).sum(axis=1) / len(query_ids))
    extreme = int(np.sum(null_statistics >= observed - 1e-15))
    return {
        "p_value": (extreme + 1) / (permutations + 1),
        "extreme_count": extreme,
        "permutations": permutations,
        "seed": seed,
        "two_sided": True,
        "plus_one_correction": True,
    }


def query_bootstrap_ci(differences, *, replicates=10_000, seed=20261007):
    values = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(values), size=(replicates, len(values)))
    means = values[samples].mean(axis=1)
    return {
        "lower": float(np.percentile(means, 2.5)),
        "upper": float(np.percentile(means, 97.5)),
        "replicates": replicates,
        "seed": seed,
    }


def query_sign_flip_test(differences, *, permutations=50_000, seed=20261008):
    values = np.asarray(differences, dtype=float)
    observed = abs(float(values.mean()))
    rng = np.random.default_rng(seed)
    signs = rng.integers(0, 2, size=(permutations, len(values)), dtype=np.int8)
    signs = signs * 2 - 1
    null_statistics = np.abs((signs * values).mean(axis=1))
    extreme = int(np.sum(null_statistics >= observed - 1e-15))
    return {
        "p_value": (extreme + 1) / (permutations + 1),
        "extreme_count": extreme,
        "permutations": permutations,
        "seed": seed,
        "two_sided": True,
        "plus_one_correction": True,
    }


def holm_adjust(p_values):
    """Return Holm-adjusted p-values in the original key order."""
    ordered = sorted(p_values, key=p_values.get)
    count = len(ordered)
    adjusted = {}
    running = 0.0
    for rank, key in enumerate(ordered):
        candidate = min(1.0, (count - rank) * p_values[key])
        running = max(running, candidate)
        adjusted[key] = running
    return {key: adjusted[key] for key in p_values}


def positive_rank(ranking, document_id):
    try:
        return ranking.index(document_id) + 1
    except ValueError:
        return ">100"
