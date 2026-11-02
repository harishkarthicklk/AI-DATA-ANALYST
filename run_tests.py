"""Run local smoke tests for dataset loading, spatial analysis, and reports.

Run from the project root with the project's Python environment:
    python run_tests.py
"""
from __future__ import annotations

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def main() -> int:
    import sys
    sys.path.insert(0, str(BASE_DIR / "server"))

    from core import get_dataframe
    from session import session
    from spatial_sequential_server import show_columns, run_dbscan, run_prefixspan, run_gsp, run_hmm
    from report_server import export_pdf_report

    session.set_dataset("datasets/spatial_data.csv")
    df = get_dataframe()
    assert not df.empty, "Spatial dataset is empty"
    assert {"latitude", "longitude"}.issubset(df.columns), "Coordinate columns are missing"

    schema = show_columns()
    assert schema.get("status") == "success", f"Schema inspection failed: {schema}"

    dbscan = run_dbscan(
        latitude_col="latitude",
        longitude_col="longitude",
        eps_km=1.0,
        min_samples=5,
    )
    assert dbscan.get("status") == "success", f"DBSCAN failed: {dbscan}"

    report_a = export_pdf_report(
        title="Spatial coordinates smoke test",
        user_request="Create a spatial report using latitude and longitude",
    )
    report_b = export_pdf_report(
        title="Pollution correlation smoke test",
        user_request="Create a correlation report for air_pollution_index and house_value",
    )
    assert report_a.get("status") == "success", f"First report failed: {report_a}"
    assert report_b.get("status") == "success", f"Second report failed: {report_b}"
    assert report_a.get("path") != report_b.get("path"), "Reports reused the same output path"
    assert report_a.get("request") != report_b.get("request"), "Report focus did not change"
    for result in (report_a, report_b):
        assert Path(result["path"]).exists(), f"Report file was not created: {result['path']}"

    # Regression test for serialized event paths in sequential_data.csv.
    # The event list is repeated across four time-step rows per sequence, so
    # the mining helper must decode it and count each sequence_id only once.
    session.set_dataset("datasets/sequential_data.csv")
    prefixspan = run_prefixspan(
        entity_col="sequence_id",
        event_col="user_event_path",
        time_col="time_step",
        min_support=2,
    )
    assert prefixspan.get("status") == "success", f"PrefixSpan failed: {prefixspan}"
    assert prefixspan.get("sequence_count") == 300, f"Unexpected sequence count: {prefixspan}"
    assert prefixspan.get("minimum_support_count") == 2, f"Unexpected support threshold: {prefixspan}"
    assert len(prefixspan.get("patterns", [])) <= 5, "PrefixSpan output exceeds five patterns"
    assert prefixspan.get("method", "").startswith("PrefixSpan"), f"Wrong mining method: {prefixspan}"

    gsp = run_gsp(
        entity_col="sequence_id",
        event_col="user_event_path",
        time_col="time_step",
        min_support=2,
    )
    hmm = run_hmm(
        entity_col="sequence_id",
        event_col="user_event_path",
        time_col="time_step",
    )
    assert gsp.get("status") == "success", f"GSP failed: {gsp}"
    assert hmm.get("status") == "success", f"HMM transition summary failed: {hmm}"

    print(json.dumps({
        "status": "success",
        "dataset_shape": list(df.shape),
        "schema_status": schema.get("status"),
        "dbscan_status": dbscan.get("status"),
        "dbscan_clusters": dbscan.get("clusters_found"),
        "report_1": report_a.get("path"),
        "report_2": report_b.get("path"),
        "reports_are_distinct": report_a.get("path") != report_b.get("path"),
        "prefixspan_status": prefixspan.get("status"),
        "prefixspan_sequence_count": prefixspan.get("sequence_count"),
        "prefixspan_patterns_returned": len(prefixspan.get("patterns", [])),
        "gsp_status": gsp.get("status"),
        "hmm_transition_summary_status": hmm.get("status"),
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
