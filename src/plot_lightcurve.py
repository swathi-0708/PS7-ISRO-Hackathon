import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from config import STAR_CONFIG
from pipeline import load_fits, clean_and_normalize


def main():
    parser = argparse.ArgumentParser(description="Plot light curve for a star.")
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

    print(f"Plotting light curve for star: {star_name}")
    try:
        # Load FITS
        fits_data = load_fits(fits_path)
        time = fits_data["time"]
        flux = fits_data["flux"]
        quality = fits_data["quality"]

        # Clean and normalize
        clean_time, clean_flux = clean_and_normalize(time, flux, quality)

        # ----------------------------------------------------
        # Plot
        # ----------------------------------------------------
        plt.figure(figsize=(12, 5))

        plt.plot(
            clean_time,
            clean_flux,
            ".",
            markersize=1,
        )

        plt.xlabel("Time (Days)")
        plt.ylabel("Normalized Flux")
        plt.title("TESS Light Curve")
        plt.grid(alpha=0.3)

        # Create results folder if needed
        Path("results").mkdir(exist_ok=True)

        plt.savefig("results/lightcurve.png", dpi=300)
        plt.show()

        print("Light curve saved to results/lightcurve.png")

    except Exception as e:
        print(f"Error processing star {star_name}: {e}")


if __name__ == "__main__":
    main()