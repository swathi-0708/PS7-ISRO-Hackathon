from pathlib import Path

import numpy as np
from astropy.io import fits
from wotan import flatten
from transitleastsquares import transitleastsquares


def main():

    # ----------------------------------------------------
    # FITS file location
    # ----------------------------------------------------

    fits_path = Path(
        "data/raw/sector_1/"
        "tess2018206045859-s0001-0000000025155310-0120-s/"
        "tess2018206045859-s0001-0000000025155310-0120-s_lc.fits"
    )

    print("Opening FITS file...")

    # ----------------------------------------------------
    # Read FITS
    # ----------------------------------------------------

    with fits.open(fits_path) as hdul:

        data = hdul[1].data

        time = data["TIME"]
        flux = data["PDCSAP_FLUX"]
        quality = data["QUALITY"]

    # ----------------------------------------------------
    # Remove bad observations
    # ----------------------------------------------------

    mask = (
        np.isfinite(time)
        & np.isfinite(flux)
        & (quality == 0)
    )

    time = time[mask]
    flux = flux[mask]

    print(f"Using {len(time)} good observations")

    # ----------------------------------------------------
    # Normalize
    # ----------------------------------------------------

    flux = flux / np.median(flux)

    # ----------------------------------------------------
    # Detrend using Wotan
    # ----------------------------------------------------

    print("Detrending light curve...")

    flat_flux = flatten(
        time,
        flux,
        method="biweight",
        window_length=0.5,
    )

    # Normalize again after detrending
    flat_flux = flat_flux / np.median(flat_flux)

    print("Detrending complete.")

    # ----------------------------------------------------
    # Transit Least Squares
    # ----------------------------------------------------

    print("\nStarting Transit Least Squares search...\n")

    model = transitleastsquares(time, flat_flux)

    results = model.power(
        show_progress_bar=True
    )

    print("\n===================================")
    print("Transit Least Squares Results")
    print("===================================")

    print(f"Best Period      : {results.period:.6f} days")
    print(f"Transit Depth    : {results.depth:.8f}")
    print(f"Transit Duration : {results.duration:.6f} days")
    print(f"SDE              : {results.SDE:.2f}")

    print("\nTLS completed successfully.")


if __name__ == "__main__":
    main()