#!/usr/bin/env python
"""
run_weekly.py — Script di convenienza per l'aggiornamento settimanale

Utilizzo:
    cd gold_model
    python run_weekly.py                     # pipeline completa
    python run_weekly.py --skip-download     # usa dati FRED/yfinance esistenti
    python run_weekly.py --skip-rebuild      # usa feature già calcolate
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.pipeline.update_pipeline import main
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gold Trend — Pipeline Settimanale")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-rebuild",  action="store_true")
    args = parser.parse_args()
    main(skip_download=args.skip_download, skip_rebuild=args.skip_rebuild)
