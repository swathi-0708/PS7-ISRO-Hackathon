import argparse
from pathlib import Path
from config import STAR_CONFIG
from pipeline import load_fits, clean_and_normalize, detrend, run_tls


def main():
    parser = argparse.ArgumentParser(description="Detect transits for a star using TLS.")
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

    print(f"Detecting transits for star: {star_name}")
    try:
        # Load FITS
        print("Opening FITS file...")
        fits_data = load_fits(fits_path)
        time = fits_data["time"]
        flux = fits_data["flux"]
        quality = fits_data["quality"]

        # Clean and normalize
        clean_time, clean_flux = clean_and_normalize(time, flux, quality)
        print(f"Using {len(clean_time)} good observations")

        # Detrend
        print("Detrending light curve...")
        flat_flux = detrend(clean_time, clean_flux, window_length=0.5, method="biweight")
        print("Detrending complete.")

        # Transit Least Squares
        print("\nStarting Transit Least Squares search...\n")
        results = run_tls(
            clean_time,
            flat_flux,
            period_min=0.5,
            period_max=15.0,
            show_progress_bar=False
        )

        print("\n===================================")
        print("Transit Least Squares Results")
        print("===================================")
        print(f"Best Period      : {results.period:.6f} days")
        print(f"Transit Depth    : {results.depth:.8f}")
        print(f"Transit Duration : {results.duration:.6f} days")
        print(f"SDE              : {results.SDE:.2f}")
        print("\nTLS completed successfully.")

    except Exception as e:
        print(f"Error processing star {star_name}: {e}")


if __name__ == "__main__":
    main()