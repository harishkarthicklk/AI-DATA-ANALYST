
from __future__ import annotations

from collections import Counter, defaultdict
import ast
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastmcp import FastMCP
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler

try:
    from .core import get_dataframe, records
except ImportError:
    from core import get_dataframe, records


mcp = FastMCP("AnalyticsAI Spatial Sequential Server")


# ============================================================
# SHARED HELPERS
# ============================================================

def get_current_dataframe() -> pd.DataFrame:
    """
    Get the currently loaded dataset.

    This function relies on core.py and session.py. If the
    MCP servers use separate in-memory sessions, core.py must
    load the dataset from a shared file path or shared session.
    """
    df = get_dataframe()

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "get_dataframe() did not return a Pandas DataFrame."
        )

    if df.empty:
        raise ValueError(
            "The loaded dataset is empty. Load a dataset first."
        )

    return df


def require(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    missing = [col for col in cols if col not in df.columns]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}. "
            f"Available columns: {list(map(str, df.columns))}"
        )

    result = df.dropna(subset=cols).copy()

    if result.empty:
        raise ValueError(
            f"No rows remain after removing missing values in {cols}."
        )

    return result


def validate_coordinates(
    df: pd.DataFrame,
    latitude_col: str,
    longitude_col: str,
) -> pd.DataFrame:
    """Validate and clean geographic coordinates."""

    result = require(
        df,
        [latitude_col, longitude_col],
    )

    result[latitude_col] = pd.to_numeric(
        result[latitude_col],
        errors="coerce",
    )

    result[longitude_col] = pd.to_numeric(
        result[longitude_col],
        errors="coerce",
    )

    result = result.dropna(
        subset=[latitude_col, longitude_col]
    ).copy()

    result = result[
        result[latitude_col].between(-90, 90)
        & result[longitude_col].between(-180, 180)
    ].copy()

    if result.empty:
        raise ValueError(
            "No valid latitude/longitude coordinates were found."
        )

    return result


def projected_coordinates(
    df: pd.DataFrame,
    latitude_col: str,
    longitude_col: str,
) -> np.ndarray:
    """
    Convert latitude and longitude into approximate kilometre
    coordinates. Suitable for local-area spatial analysis.
    """

    coordinates = df[
        [latitude_col, longitude_col]
    ].to_numpy(dtype=float)

    latitude_radians = np.radians(coordinates[:, 0])
    longitude_radians = np.radians(coordinates[:, 1])

    mean_latitude = latitude_radians.mean()

    x = longitude_radians * np.cos(mean_latitude)
    y = latitude_radians

    return np.column_stack((x, y)) * 6371.0


def safe_records(df: pd.DataFrame) -> list[dict]:
    """Convert a DataFrame into JSON-friendly records."""

    result = records(df)

    if isinstance(result, list):
        return result

    return df.to_dict(orient="records")


def error_result(exc: Exception) -> dict:
    return {
        "status": "error",
        "error_type": type(exc).__name__,
        "message": str(exc),
    }


# ============================================================
# DATASET INSPECTION
# ============================================================

@mcp.tool
def show_columns() -> dict:
    """Show columns and data types of the currently loaded dataset."""

    try:
        df = get_current_dataframe()

        return {
            "status": "success",
            "rows": int(len(df)),
            "column_count": int(len(df.columns)),
            "columns": [
                {
                    "name": str(col),
                    "dtype": str(df[col].dtype),
                }
                for col in df.columns
            ],
        }

    except Exception as exc:
        return error_result(exc)


# ============================================================
# DBSCAN SPATIAL CLUSTERING
# ============================================================

@mcp.tool
def run_dbscan(
    latitude_col: str,
    longitude_col: str,
    eps_km: float = 1.0,
    min_samples: int = 5,
) -> dict:
    """
    Group nearby geographic points using DBSCAN.

    eps_km: Maximum neighborhood distance in kilometres.
    min_samples: Minimum number of nearby observations required
    to form a dense region.
    """

    try:
        if eps_km <= 0:
            raise ValueError("eps_km must be greater than zero.")

        if min_samples < 2:
            raise ValueError("min_samples must be at least 2.")

        df = validate_coordinates(
            get_current_dataframe(),
            latitude_col,
            longitude_col,
        )

        xy = projected_coordinates(
            df,
            latitude_col,
            longitude_col,
        )

        labels = DBSCAN(
            eps=float(eps_km),
            min_samples=int(min_samples),
        ).fit_predict(xy)

        cluster_counts = Counter(
            int(label)
            for label in labels
            if label != -1
        )

        samples = []

        for i in range(min(5, len(df))):
            samples.append(
                {
                    "latitude": float(
                        df.iloc[i][latitude_col]
                    ),
                    "longitude": float(
                        df.iloc[i][longitude_col]
                    ),
                    "cluster": int(labels[i]),
                }
            )

        return {
            "status": "success",
            "method": "DBSCAN",
            "distance_unit": "kilometres",
            "eps_km": float(eps_km),
            "min_samples": int(min_samples),
            "total_records": int(len(labels)),
            "clusters_found": int(len(cluster_counts)),
            "noise_points": int((labels == -1).sum()),
            "cluster_sizes": dict(cluster_counts),
            "sample_assignments": samples,
        }

    except Exception as exc:
        return error_result(exc)


# ============================================================
# GEOGRAPHICALLY WEIGHTED REGRESSION
# ============================================================

@mcp.tool
def run_gwr(
    latitude_col: str,
    longitude_col: str,
    target_col: str,
    bandwidth_km: float = 5.0,
) -> dict:
    """
    Lightweight geographically weighted linear regression.

    Fits a locally weighted model at each observation.
    """

    try:
        if bandwidth_km <= 0:
            raise ValueError(
                "bandwidth_km must be greater than zero."
            )

        df = validate_coordinates(
            get_current_dataframe(),
            latitude_col,
            longitude_col,
        )

        df[target_col] = pd.to_numeric(
            df[target_col],
            errors="coerce",
        )

        df = df.dropna(subset=[target_col]).copy()

        if len(df) < 4:
            raise ValueError(
                "At least four observations are required for GWR."
            )

        y = df[target_col].to_numpy(dtype=float)

        # Predictor variables: intercept, latitude and longitude.
        X = np.column_stack(
            [
                np.ones(len(df)),
                df[[latitude_col, longitude_col]].to_numpy(
                    dtype=float
                ),
            ]
        )

        xy = projected_coordinates(
            df,
            latitude_col,
            longitude_col,
        )

        distance_matrix = cdist(xy, xy)

        predictions = []
        bandwidth = float(bandwidth_km)

        for i in range(len(df)):
            weights = np.exp(
                -(distance_matrix[i] ** 2)
                / (2 * bandwidth ** 2)
            )

            # Weighted least squares without creating a large
            # diagonal weight matrix.
            weighted_X = X * weights[:, None]

            beta = np.linalg.pinv(
                X.T @ weighted_X
            ) @ (
                X.T @ (weights * y)
            )

            predictions.append(float(X[i] @ beta))

        output = df[
            [latitude_col, longitude_col, target_col]
        ].copy()

        output["gwr_predicted"] = predictions
        output["residual"] = (
            output[target_col] - output["gwr_predicted"]
        )

        return {
            "status": "success",
            "method": "Lightweight GWR",
            "target_column": target_col,
            "bandwidth_km": bandwidth,
            "rows_processed": int(len(output)),
            "results_returned": min(100, len(output)),
            "results": safe_records(output.head(5)),
            "note": (
                "This is a simplified local weighted regression "
                "implementation, not a full MGWR implementation."
            ),
        }

    except Exception as exc:
        return error_result(exc)


# ============================================================
# SPATIAL CO-LOCATION
# ============================================================

@mcp.tool
def run_spatial_colocation(
    latitude_col: str,
    longitude_col: str,
    category_col: str,
    distance_threshold_km: float = 1.0,
) -> dict:
    """
    Count pairs of observations from different categories
    that fall within the specified distance.
    """

    try:
        if distance_threshold_km < 0:
            raise ValueError(
                "distance_threshold_km cannot be negative."
            )

        df = validate_coordinates(
            get_current_dataframe(),
            latitude_col,
            longitude_col,
        )

        df = require(df, [category_col])

        xy = projected_coordinates(
            df,
            latitude_col,
            longitude_col,
        )

        distance_matrix = cdist(xy, xy)
        categories = df[category_col].astype(str).to_numpy()
        unique_categories = list(pd.unique(categories))

        results = {}

        for i, category_a in enumerate(unique_categories):
            for category_b in unique_categories[i + 1:]:
                indices_a = np.where(
                    categories == category_a
                )[0]

                indices_b = np.where(
                    categories == category_b
                )[0]

                count = int(
                    (
                        distance_matrix[
                            np.ix_(indices_a, indices_b)
                        ] <= float(distance_threshold_km)
                    ).sum()
                )

                results[
                    f"{category_a} <-> {category_b}"
                ] = count

        return {
            "status": "success",
            "distance_threshold_km": float(
                distance_threshold_km
            ),
            "category_count": len(unique_categories),
            "colocation_counts": results,
        }

    except Exception as exc:
        return error_result(exc)


# ============================================================
# SPATIAL INTERPOLATION
# ============================================================

@mcp.tool
def run_kriging_interpolation(
    value_col: str,
    latitude_col: str,
    longitude_col: str,
    target_lat: float,
    target_lon: float,
) -> dict:
    """
    Estimate a value at a target location using inverse-distance
    weighting. Despite the tool name, this is not true Kriging.
    """

    try:
        if not -90 <= target_lat <= 90:
            raise ValueError(
                "Target latitude must be between -90 and 90."
            )

        if not -180 <= target_lon <= 180:
            raise ValueError(
                "Target longitude must be between -180 and 180."
            )

        df = validate_coordinates(
            get_current_dataframe(),
            latitude_col,
            longitude_col,
        )

        df[value_col] = pd.to_numeric(
            df[value_col],
            errors="coerce",
        )

        df = df.dropna(subset=[value_col]).copy()

        if df.empty:
            raise ValueError(
                "No valid numeric values are available to interpolate."
            )

        xy = projected_coordinates(
            df,
            latitude_col,
            longitude_col,
        )

        target_df = pd.DataFrame(
            {
                latitude_col: [target_lat],
                longitude_col: [target_lon],
            }
        )

        target_xy = projected_coordinates(
            target_df,
            latitude_col,
            longitude_col,
        )

        distances = cdist(xy, target_xy).ravel()
        values = df[value_col].to_numpy(dtype=float)

        exact_matches = np.where(distances < 1e-9)[0]

        if len(exact_matches) > 0:
            prediction = float(
                values[exact_matches].mean()
            )
        else:
            weights = 1.0 / np.maximum(
                distances,
                1e-9,
            ) ** 2

            prediction = float(
                np.sum(weights * values) / np.sum(weights)
            )

        return {
            "status": "success",
            "method": "Inverse-distance weighting (IDW)",
            "target_coordinates": {
                "latitude": float(target_lat),
                "longitude": float(target_lon),
            },
            "value_column": value_col,
            "interpolated_value": prediction,
            "note": (
                "This uses IDW interpolation, not geostatistical "
                "Kriging with a fitted variogram."
            ),
        }

    except Exception as exc:
        return error_result(exc)


# ============================================================
# SPATIAL OUTLIER DETECTION
# ============================================================

@mcp.tool
def run_spatial_outlier_detection(
    latitude_col: str,
    longitude_col: str,
    attribute_col: str,
    k_neighbors: int = 5,
    threshold_z: float = 2.5,
) -> dict:
    """
    Flag observations whose attribute value differs substantially
    from the values of their nearest geographic neighbors.
    """

    try:
        if k_neighbors < 1:
            raise ValueError(
                "k_neighbors must be at least 1."
            )

        if threshold_z <= 0:
            raise ValueError(
                "threshold_z must be greater than zero."
            )

        df = validate_coordinates(
            get_current_dataframe(),
            latitude_col,
            longitude_col,
        )

        df[attribute_col] = pd.to_numeric(
            df[attribute_col],
            errors="coerce",
        )

        df = df.dropna(subset=[attribute_col]).copy()

        if len(df) < 2:
            raise ValueError(
                "At least two observations are required."
            )

        xy = projected_coordinates(
            df,
            latitude_col,
            longitude_col,
        )

        distances = cdist(xy, xy)
        values = df[attribute_col].to_numpy(dtype=float)

        k = min(int(k_neighbors), len(df) - 1)
        outliers = []

        for i in range(len(df)):
            neighbor_indices = np.argsort(
                distances[i]
            )[1:k + 1]

            neighbor_values = values[neighbor_indices]
            neighbor_mean = float(neighbor_values.mean())
            neighbor_std = float(neighbor_values.std())

            if neighbor_std > 0:
                z_score = abs(
                    float(values[i]) - neighbor_mean
                ) / neighbor_std
            else:
                z_score = (
                    float("inf")
                    if values[i] != neighbor_mean
                    else 0.0
                )

            if z_score > threshold_z:
                outliers.append(
                    {
                        "row_index": str(df.index[i]),
                        "latitude": float(
                            df.iloc[i][latitude_col]
                        ),
                        "longitude": float(
                            df.iloc[i][longitude_col]
                        ),
                        "value": float(values[i]),
                        "neighbors_mean": neighbor_mean,
                        "z_score": (
                            z_score
                            if np.isfinite(z_score)
                            else None
                        ),
                    }
                )

        return {
            "status": "success",
            "attribute_column": attribute_col,
            "neighbors_used": k,
            "threshold_z": float(threshold_z),
            "total_records_checked": int(len(df)),
            "total_outliers_found": len(outliers),
            "outliers": outliers[:5],
            "outliers_truncated": len(outliers) > 5,
        }

    except Exception as exc:
        return error_result(exc)


# ============================================================
# SEQUENTIAL DATA HELPERS
# ============================================================

def sequences(
    df: pd.DataFrame,
    entity_col: str,
    event_col: str,
    time_col: str | None = None,
) -> list[list[str]]:
    columns = [entity_col, event_col]

    if time_col:
        columns.append(time_col)

    df = require(df, columns)

    if time_col:
        df = df.sort_values([entity_col, time_col])

    output = []
    for _, group in df.groupby(entity_col, sort=False):
        values = group[event_col].tolist()
        # Some event-path datasets repeat a serialized list (e.g. "['Home',
        # 'Search', 'Cart']") on every time-step row. Decode it and align one
        # event per row when the path length matches the entity's observations.
        if values and isinstance(values[0], str):
            try:
                parsed = ast.literal_eval(values[0])
            except (ValueError, SyntaxError):
                parsed = None
            if isinstance(parsed, (list, tuple)) and all(isinstance(x, (str, int, float)) for x in parsed):
                if len(parsed) == len(group):
                    values = list(parsed)
        cleaned = [str(value).strip() for value in values if pd.notna(value) and str(value).strip()]
        if cleaned:
            output.append(cleaned)
    return output


# ============================================================
# PREFIXSPAN SEQUENTIAL PATTERN MINING
# ============================================================

@mcp.tool
def run_prefixspan(
    entity_col: str,
    event_col: str,
    min_support: float = 0.2,
    time_col: str = "",
) -> dict:
    """Mine frequent ordered subsequences with PrefixSpan-style projection.

    Each entity contributes at most once to a pattern's support. If an event
    column contains a serialized list repeated across an entity's time rows,
    ``sequences`` decodes the list and uses it as that entity's event sequence.
    ``min_support`` in (0, 1] is a relative support fraction; values > 1 are
    interpreted as an absolute sequence count.
    """
    try:
        if not np.isfinite(float(min_support)) or float(min_support) <= 0:
            raise ValueError("min_support must be greater than zero.")

        df = get_current_dataframe()
        seqs = sequences(df, entity_col, event_col, time_col or None)
        # PrefixSpan operates on ordered sequences of discrete event symbols.
        seqs = [tuple(str(item).strip() for item in seq if str(item).strip()) for seq in seqs]
        seqs = [seq for seq in seqs if seq]
        n = len(seqs)
        if n == 0:
            raise ValueError("No non-empty entity sequences were found.")

        support_value = float(min_support)
        min_count = (max(1, int(np.ceil(support_value * n)))
                     if support_value <= 1 else int(np.ceil(support_value)))
        if min_count > n:
            return {
                "status": "success",
                "method": "PrefixSpan (projected sequential pattern mining)",
                "sequence_count": n,
                "minimum_support": support_value,
                "minimum_support_count": min_count,
                "patterns": [],
                "patterns_truncated": False,
                "message": "No patterns meet the requested minimum support; the support count exceeds the number of sequences.",
            }

        counts: Counter[tuple[str, ...]] = Counter()

        # Projected database rows contain the remaining suffix for each input
        # sequence. The earliest occurrence of a prefix leaves the largest
        # possible suffix, so each entity needs only one projected suffix.
        def grow(prefix: tuple[str, ...], projected: list[tuple[str, ...]]) -> None:
            extension_entities: dict[str, list[tuple[str, ...]]] = defaultdict(list)
            for suffix in projected:
                seen_items = set()
                for item in suffix:
                    if item not in seen_items:
                        extension_entities[item].append(suffix)
                        seen_items.add(item)

            for item, suffixes in extension_entities.items():
                if len(suffixes) < min_count:
                    continue
                pattern = prefix + (item,)
                counts[pattern] = len(suffixes)
                next_projected = []
                for suffix in suffixes:
                    # Project after the first matching extension item.
                    idx = suffix.index(item)
                    next_projected.append(suffix[idx + 1:])
                if any(next_projected):
                    grow(pattern, next_projected)

        grow((), seqs)
        patterns = [
            {"pattern": list(pattern), "support_count": int(count), "support": float(count / n)}
            for pattern, count in counts.items()
            if count >= min_count
        ]
        patterns.sort(key=lambda item: (-item["support_count"], len(item["pattern"]), item["pattern"]))

        return {
            "status": "success",
            "method": "PrefixSpan (projected sequential pattern mining)",
            "sequence_count": n,
            "minimum_support": support_value,
            "minimum_support_count": min_count,
            "patterns": patterns[:5],
            "patterns_truncated": len(patterns) > 5,
            "total_frequent_patterns": len(patterns),
            "event_column": event_col,
            "entity_column": entity_col,
        }
    except Exception as exc:
        return error_result(exc)


# ============================================================
# GSP-LIKE SEQUENTIAL PATTERN MINING
# ============================================================

@mcp.tool
def run_gsp(
    entity_col: str,
    event_col: str,
    min_support: float = 0.2,
    time_col: str = "",
) -> dict:
    """
    Count frequent consecutive event pairs.

    This is a simplified consecutive-pair miner, not a complete
    Generalized Sequential Pattern (GSP) implementation.
    """

    try:
        if min_support <= 0:
            raise ValueError("min_support must be greater than zero.")

        seqs = sequences(
            get_current_dataframe(),
            entity_col,
            event_col,
            time_col or None,
        )

        n = len(seqs)

        if n == 0:
            raise ValueError("No entity sequences were found.")

        counts = Counter()

        for sequence in seqs:
            # Count a pair at most once per entity.
            entity_pairs = set(
                tuple(sequence[i:i + 2])
                for i in range(len(sequence) - 1)
            )

            counts.update(entity_pairs)

        threshold = max(1, int(np.ceil(min_support * n))) if min_support <= 1 else int(np.ceil(min_support))

        patterns = [
            {
                "pattern": list(pattern),
                "support_count": int(count),
                "support": float(count / n),
            }
            for pattern, count in counts.items()
            if count >= threshold
        ]

        patterns.sort(
            key=lambda item: -item["support_count"]
        )

        return {
            "status": "success",
            "method": "Simplified consecutive-pair mining",
            "sequence_count": n,
            "minimum_support": float(min_support),
            "minimum_support_count": threshold,
            "patterns": patterns[:5],
            "patterns_truncated": len(patterns) > 5,
        }

    except Exception as exc:
        return error_result(exc)


# ============================================================
# HMM-STYLE TRANSITION SUMMARY
# ============================================================

@mcp.tool
def run_hmm(
    entity_col: str,
    event_col: str,
    time_col: str = "",
) -> dict:
    """
    Estimate observed event-to-event transition probabilities.

    This produces a transition summary; it does not train a
    complete hidden Markov model.
    """

    try:
        seqs = sequences(
            get_current_dataframe(),
            entity_col,
            event_col,
            time_col or None,
        )

        if not seqs:
            raise ValueError("No event sequences were found.")

        states = sorted(
            set(event for sequence in seqs for event in sequence)
        )

        transitions = defaultdict(Counter)

        for sequence in seqs:
            for current, following in zip(
                sequence,
                sequence[1:],
            ):
                transitions[current][following] += 1

        matrix = {}

        for current in states:
            total = sum(transitions[current].values())

            matrix[current] = {
                following: (
                    float(transitions[current][following] / total)
                    if total
                    else 0.0
                )
                for following in states
            }

        return {
            "status": "success",
            "states": states,
            "transition_probabilities": matrix,
            "note": (
                "Probabilities are estimated from observed events. "
                "A trained hidden Markov model is not fitted."
            ),
        }

    except Exception as exc:
        return error_result(exc)


# ============================================================
# DYNAMIC TIME WARPING
# ============================================================

@mcp.tool
def run_dtw(
    entity_col: str,
    time_col: str,
    value_col: str,
) -> dict:
    """Compare numeric sequences using dynamic time warping."""

    try:
        df = require(
            get_current_dataframe(),
            [entity_col, time_col, value_col],
        )

        df[value_col] = pd.to_numeric(
            df[value_col],
            errors="coerce",
        )

        df = df.dropna(subset=[value_col]).copy()

        sequences_by_entity = {
            entity: group.sort_values(time_col)[
                value_col
            ].to_numpy(dtype=float)
            for entity, group in df.groupby(entity_col)
        }

        keys = list(sequences_by_entity)
        output = []

        for i, entity_a in enumerate(keys):
            for entity_b in keys[i + 1:]:
                x = sequences_by_entity[entity_a]
                y = sequences_by_entity[entity_b]

                dp = np.full(
                    (len(x) + 1, len(y) + 1),
                    np.inf,
                )

                dp[0, 0] = 0.0

                for row in range(1, len(x) + 1):
                    for col in range(1, len(y) + 1):
                        cost = abs(
                            x[row - 1] - y[col - 1]
                        )

                        dp[row, col] = cost + min(
                            dp[row - 1, col],
                            dp[row, col - 1],
                            dp[row - 1, col - 1],
                        )

                output.append(
                    {
                        "entity_a": str(entity_a),
                        "entity_b": str(entity_b),
                        "dtw_distance": float(dp[-1, -1]),
                    }
                )

        output.sort(
            key=lambda item: item["dtw_distance"]
        )

        return {
            "status": "success",
            "entity_count": len(keys),
            "pairwise_distances_returned": min(5, len(output)),
            "pairwise_distances_total": len(output),
            "pairwise_distances": output[:5],
        }

    except Exception as exc:
        return error_result(exc)


# ============================================================
# TRAJECTORY CLUSTERING
# ============================================================

@mcp.tool
def run_trajectory_clustering(
    entity_col: str,
    latitude_col: str,
    longitude_col: str,
    n_clusters: int = 3,
) -> dict:
    """
    Cluster entities by their mean latitude and longitude.

    This clusters mean locations, not the full movement paths.
    """

    try:
        if n_clusters < 1:
            raise ValueError(
                "n_clusters must be at least 1."
            )

        df = validate_coordinates(
            get_current_dataframe(),
            latitude_col,
            longitude_col,
        )

        df = require(df, [entity_col])

        features = df.groupby(entity_col)[
            [latitude_col, longitude_col]
        ].mean()

        if features.empty:
            raise ValueError("No entity locations were found.")

        k = min(int(n_clusters), len(features))

        if k == 1:
            labels = np.zeros(len(features), dtype=int)
        else:
            scaled = StandardScaler().fit_transform(features)

            labels = KMeans(
                n_clusters=k,
                n_init=10,
                random_state=42,
            ).fit_predict(scaled)

        output = features.copy()
        output["cluster"] = labels

        return {
            "status": "success",
            "method": "KMeans on mean entity coordinates",
            "clusters": int(k),
            "entities_processed": int(len(features)),
            "assignments": safe_records(
                output.reset_index().head(5)
            ),
            "assignments_total": int(len(output)),
            "assignments_truncated": len(output) > 5,
        }

    except Exception as exc:
        return error_result(exc)


# ============================================================
# START MCP SERVER
# ============================================================

if __name__ == "__main__":
    mcp.run()