import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from config import STAR_CONFIG
from pipeline import load_fits, clean_and_normalize, detrend

def main():
    parser = argparse.ArgumentParser(description="Preprocess light curve for a star.")
    parser.add_argument("--star", type=str, default="TIC_25155310", help="Star name to process")
    args = parser.parse_args()

    star_name = args.star
    if star_name not in STAR_CONFIG:
        print(f"Error: Star '{star_name}' not found in STAR_CONFIG.")
        return

    star_info = STAR_CONFIG[star_name]
    fits_path = Path(star_info["fits_path"])

    print(f"Preprocessing star: {star_name}")
    try:
        # Load FITS
        fits_data = load_fits(fits_path)
        time = fits_data["time"]
        flux = fits_data["flux"]
        quality = fits_data["quality"]

        # Clean and normalize
        clean_time, clean_flux = clean_and_normalize(time, flux, quality)
        raw_flux = clean_flux.copy()

        # Detrend
        flat_flux = detrend(clean_time, clean_flux, window_length=0.5, method="biweight")

        # Create comparison plots
        plt.figure(figsize=(12, 8))

        # Raw light curve
        plt.subplot(2, 1, 1)
        plt.plot(
            clean_time,
            raw_flux,
            ".",
            markersize=1,
        )
        plt.title("Raw Normalized TESS Light Curve")
        plt.ylabel("Normalized Flux")
        plt.grid(alpha=0.3)

        # Detrended light curve
        plt.subplot(2, 1, 2)
        plt.plot(
            clean_time,
            flat_flux,
            ".",
            markersize=1,
        )
        plt.title("Detrended TESS Light Curve (Wotan)")
        plt.xlabel("Time (Days)")
        plt.ylabel("Normalized Flux")
        plt.grid(alpha=0.3)

        plt.tight_layout()

        # Save figure
        Path("results").mkdir(exist_ok=True)
        plt.savefig(
            "results/preprocessed_lightcurve.png",
            dpi=300
        )
        plt.show()

        print("Preprocessed light curve saved to results/preprocessed_lightcurve.png")

    except Exception as e:
        print(f"Error processing star {star_name}: {e}")

if __name__ == "__main__":
    main()