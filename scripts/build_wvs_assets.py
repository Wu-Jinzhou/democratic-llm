#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from dempo.wvs import (
    build_global_panel_config_dict,
    build_subjective_questions,
    write_global_panel_config,
    write_subjective_questions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build WVS-derived assets: filtered subjective questions and a global panel config."
    )
    parser.add_argument(
        "--wvs-csv",
        type=Path,
        default=Path("wvs/WVS_Cross-National_Wave_7_csv_v6_0.csv"),
        help="Path to the WVS wave-7 CSV.",
    )
    parser.add_argument(
        "--questions-output",
        type=Path,
        default=Path("wvs/subjective_questions.json"),
        help="Where to write the filtered subjective question set.",
    )
    parser.add_argument(
        "--panel-config-output",
        type=Path,
        default=Path("configs/panel_config_global_wvs.yaml"),
        help="Where to write the WVS-derived global panel config.",
    )
    parser.add_argument(
        "--panel-size",
        type=int,
        default=100,
        help="Target panel size for the global configuration.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.10,
        help="Relative tolerance for each demographic target in the panel config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    questions = write_subjective_questions(args.questions_output)
    config = write_global_panel_config(
        args.wvs_csv,
        args.panel_config_output,
        panel_size=args.panel_size,
        tolerance=args.tolerance,
    )

    wvs = pd.read_csv(args.wvs_csv, usecols=["B_COUNTRY_ALPHA"])
    countries = int(wvs["B_COUNTRY_ALPHA"].dropna().nunique())
    summary = {
        "question_count": len(questions),
        "country_count": countries,
        "panel_size": int(config["panel_size"]),
        "tolerance": float(config["tolerance"]),
        "question_file": str(args.questions_output),
        "panel_config_file": str(args.panel_config_output),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
