"""Read the sixty completed remote traces; do not execute tasks or fit models."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wmagentattack.clean_pairing import clean_gate
from wmagentattack.clean_trace_audit import describe_trace, summarize_traces


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    archive = args.archive.resolve()
    output = archive / "posthoc_failure_audit.json"
    if output.exists():
        raise FileExistsError("Descriptive audit already archived; do not overwrite")
    p = json.loads((archive / "preregistered_protocol.json").read_text())
    results = json.loads((archive / "results.json").read_text())
    if not (archive / "COMPLETE").is_file():
        raise ValueError("Completion sentinel missing")
    def valid_trace(row):
        path = Path(row.get("raw_trace", "")).resolve()
        return path.is_relative_to(archive / "raw") and path.is_file() and bool(json.loads(path.read_text()))
    gate = clean_gate(results, p, valid_trace)
    if not gate["scientific_result"]:
        raise ValueError("Incomplete or invalid records")
    recorded_gate = json.loads((archive / "gate.json").read_text())
    for field, value in gate.items():
        if recorded_gate.get(field) != value:
            raise ValueError(f"Gate cross-check failed: {field}")
    detail = [describe_trace(row, json.loads(Path(row["raw_trace"]).read_text())) for row in results]
    summary = summarize_traces(detail)
    summary["recomputed_gate_matches"] = True
    if not summary["raw_utility_all_agree"]:
        raise ValueError("Raw utility labels mismatch result records")
    with output.open("x") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "details"}, sort_keys=True))


if __name__ == "__main__":
    main()
