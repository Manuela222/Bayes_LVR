from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import ROOT


ABLATIONS = [
    "configs/baseline.yaml",
    "configs/extended_action.yaml",
    "configs/bayeslvr.yaml",
    "configs/bayeslvr_gated.yaml",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rolling", action="store_true")
    parser.add_argument("--max-windows", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for cfg in ABLATIONS:
        command = [sys.executable, str(ROOT / "scripts" / "train.py"), "--config", str(ROOT / cfg)]
        if args.rolling:
            command.extend(["--rolling", "--max-windows", str(args.max_windows)])
        print("Running", cfg)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
