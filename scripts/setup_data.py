from __future__ import annotations

from pathlib import Path
import shutil

from src.utils.config import ROOT


def main() -> None:
    target = ROOT / "src" / "data" / "data_price_uni_h_time.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        print(f"Dataset already available at {target}")
        return

    legacy_source = ROOT / "finai_contest_2025" / "Task_3_FinRL_DeFi" / "src" / "data_price_uni_h_time.csv"
    if legacy_source.exists():
        shutil.copy2(legacy_source, target)
        print(f"Copied dataset to {target}")
        return

    raise FileNotFoundError(
        f"Dataset not found. Expected either {target} or legacy source {legacy_source}."
    )


if __name__ == "__main__":
    main()
