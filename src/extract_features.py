import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from config import STAR_CONFIG
from pipeline import load_fits, clean_and_normalize, detrend, run_tls


def main():
    parser = argparse.ArgumentParser(description="Extract transit features for a star.")
    parser.add_argument(
        "--star", type=str, default="TIC_25155310",
        help="Star key from STAR_CONFIG to process (default: TIC_25155310)"
    )
    args = parser.parse_args()

    star_name = args.star
    if star_name not in STAR_CONFIG:
        print(f"Error: Star '{star_name}' not found in STAR_CONFIG.")
        return

    star_info = STAR_CONFIG[star_name]
    fits_path = Path(star_info["fits_path"])

    print(f"Extracting features for star: {star_name}")
    try:
        # ----------------------------------------------------
        # Load FITS
        # ----------------------------------------------------
        print("Reading FITS file...")
        fits_data = load_fits(fits_path)
        time = fits_data["time"]
        flux = fits_data["flux"]
        quality = fits_data["quality"]
        tic_id = fits_data["tic_id"]

        # ----------------------------------------------------
        # Clean, normalize, and detrend
        # ----------------------------------------------------
        clean_time, clean_flux = clean_and_normalize(time, flux, quality)
        flat_flux = detrend(clean_time, clean_flux, window_length=0.5, method="biweight")

        # ----------------------------------------------------
        # TLS
        # ----------------------------------------------------
        print("Running TLS...")
        results = run_tls(
            clean_time,
            flat_flux,
            period_min=0.5,
            period_max=15.0,
            show_progress_bar=False
        )

        # ----------------------------------------------------
        # Feature Extraction
        # ----------------------------------------------------
        duration_hours = results.duration * 24

        features = pd.DataFrame([{
            "tic_id": tic_id,
            "period_days": results.period,
            "duration_days": results.duration,
            "duration_hours": duration_hours,
            # NOTE: TLS's `depth` output is a model-fit parameter and not necessarily the literal
            # physical fractional transit depth. The physical depth will be derived properly in
            # the planned Stage 6 batman fit.
            "depth": results.depth,
            "SDE": results.SDE
        }])

        # ----------------------------------------------------
        # Save CSV
        # ----------------------------------------------------
        Path("results").mkdir(exist_ok=True)
        output_file = Path("results/features.csv")
        features.to_csv(output_file, index=False)

        print("\n==========================")
        print("Feature Extraction Complete")
        print("==========================")
        print(features)
        print(f"\nSaved to: {output_file}")

    except Exception as e:
        print(f"Error processing star {star_name}: {e}")


if __name__ == "__main__":
    main()