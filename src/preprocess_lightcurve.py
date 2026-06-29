from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from wotan import flatten

# ----------------------------------------------------
# Path to the FITS file
# ----------------------------------------------------

fits_path = Path(
    "data/raw/sector_1/"
    "tess2018206045859-s0001-0000000025155310-0120-s/"
    "tess2018206045859-s0001-0000000025155310-0120-s_lc.fits"
)

# ----------------------------------------------------
# Read the FITS file
# ----------------------------------------------------

with fits.open(fits_path) as hdul:

    data = hdul[1].data

    time = data["TIME"]
    flux = data["PDCSAP_FLUX"]
    quality = data["QUALITY"]

# ----------------------------------------------------
# Keep only good observations
# QUALITY == 0 means no known issues
# ----------------------------------------------------

mask = (
    np.isfinite(time)
    & np.isfinite(flux)
    & (quality == 0)
)

time = time[mask]
flux = flux[mask]

# ----------------------------------------------------
# Normalize the flux
# ----------------------------------------------------

flux = flux / np.median(flux)

# Save a copy before preprocessing
raw_flux = flux.copy()

# ----------------------------------------------------
# Detrend using Wotan
# Removes long-term stellar variability
# ----------------------------------------------------

flat_flux = flatten(
    time,
    flux,
    method="biweight",
    window_length=0.5
)

# ----------------------------------------------------
# Create comparison plots
# ----------------------------------------------------

plt.figure(figsize=(12, 8))

# -----------------------
# Raw light curve
# -----------------------

plt.subplot(2, 1, 1)

plt.plot(
    time,
    raw_flux,
    ".",
    markersize=1,
)

plt.title("Raw Normalized TESS Light Curve")
plt.ylabel("Normalized Flux")
plt.grid(alpha=0.3)

# -----------------------
# Detrended light curve
# -----------------------

plt.subplot(2, 1, 2)

plt.plot(
    time,
    flat_flux,
    ".",
    markersize=1,
)

plt.title("Detrended TESS Light Curve (Wotan)")
plt.xlabel("Time (Days)")
plt.ylabel("Normalized Flux")
plt.grid(alpha=0.3)

plt.tight_layout()

# ----------------------------------------------------
# Save figure
# ----------------------------------------------------

Path("results").mkdir(exist_ok=True)

plt.savefig(
    "results/preprocessed_lightcurve.png",
    dpi=300
)

plt.show()

print("Preprocessed light curve saved to results/preprocessed_lightcurve.png")