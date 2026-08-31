"""One frozen, read-only fixed-goal audit on the remote server."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wmagentattack.fixed_goal_identifiability import align_fixed_goals, fixed_goal_audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite completed analysis")
    protocol = json.loads(args.protocol.read_text())
    if protocol["status"] != "preregistered_before_v36_metrics":
        raise ValueError("protocol is not preregistered")
    data = protocol["data"]
    manifest = json.loads(Path(data["manifest"]).read_text())["rows"]
    labels = [json.loads(line) for line in Path(data["labels"]).read_text().splitlines() if line.strip()]
    rows = align_fixed_goals(manifest, labels, data)
    result = fixed_goal_audit(rows, protocol["posterior"], protocol["gate"])
    result["protocol_id"] = protocol["protocol_id"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"decision": result["decision"], "checks": result["checks"], "metrics": result["metrics"]}, sort_keys=True))


if __name__ == "__main__":
    main()
