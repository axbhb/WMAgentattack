"""Run normalization, synthetic generation, training, evaluation, and planning."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = [
    "01_normalize_agentdojo.py",
    "02_generate_synthetic_data.py",
    "03_train_world_model.py",
    "04_eval_world_model.py",
    "05_run_attack_planner.py",
    "06_eval_attack_and_defense.py",
]


def main():
    for script in SCRIPTS:
        print(f"\n=== {script} ===", flush=True)
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / script)],
            cwd=ROOT,
            check=True,
        )


if __name__ == "__main__":
    main()
