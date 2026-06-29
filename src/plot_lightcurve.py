from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Path to the FITS file
fits_path = Path(
    "data/raw/sector_1/"
    "tess2018206045859-s0001-0000000025155310-0120-s/"
    "tess2018206045859-s0001-0000000025155310-0120-s_lc.fits"
)
# Open the FITS file
with fits.open(fits_path) as hdul:

    # The light curve is stored in HDU 1
    data = hdul[1].data

    # Extract columns
    time = data["TIME"]
    flux = data["PDCSAP_FLUX"]
    quality = data["QUALITY"]

# ----------------------------------------------------
# Keep only good-quality observations
# QUALITY == 0 means no known issues
# ----------------------------------------------------

mask = (
    np.isfinite(time)
    & np.isfinite(flux)
    & (quality == 0)
)

time = time[mask]
flux = flux[mask]

# Normalize the flux
flux = flux / np.median(flux)

# ----------------------------------------------------
# Plot
# ----------------------------------------------------

plt.figure(figsize=(12,5))

plt.plot(
    time,
    flux,
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