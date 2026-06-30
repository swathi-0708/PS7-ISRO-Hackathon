from pathlib import Path
from astropy.io import fits
import numpy as np
import wotan
from transitleastsquares import transitleastsquares
from lightkurve.utils import TessQualityFlags

def load_fits(fits_path: Path) -> dict:
    """
    Opens FITS file and returns a dictionary with keys:
    'time', 'flux', 'quality', 'tic_id'.
    """
    fits_path = Path(fits_path)
    try:
        with fits.open(fits_path) as hdul:
            header = hdul[0].header
            data = hdul[1].data
            tic_id = header["TICID"]
            time = data["TIME"]
            flux = data["PDCSAP_FLUX"]
            quality = data["QUALITY"]
            return {
                "time": time,
                "flux": flux,
                "quality": quality,
                "tic_id": tic_id
            }
    except FileNotFoundError as e:
        raise FileNotFoundError(f"FITS file not found at the specified path: {fits_path}") from e

def clean_and_normalize(time, flux, quality) -> tuple[np.ndarray, np.ndarray]:
    """
    Applies a quality bitmask filter using standard TESS/Kepler default quality bitmask,
    drops non-finite time/flux, and normalizes flux by its median.
    Prints the number of points removed by the quality filter and NaN filter separately.
    """
    nan_mask = ~(np.isfinite(time) & np.isfinite(flux))
    quality_mask = (quality & TessQualityFlags.DEFAULT_BITMASK) != 0

    num_removed_nan = np.sum(nan_mask)
    num_removed_quality = np.sum(quality_mask)

    print(f"Points removed by quality filter: {num_removed_quality}")
    print(f"Points removed by NaN filter: {num_removed_nan}")

    valid_mask = (~nan_mask) & (~quality_mask)
    clean_time = time[valid_mask]
    clean_flux = flux[valid_mask]

    clean_flux = clean_flux / np.median(clean_flux)
    return clean_time, clean_flux

def detrend(time, flux, window_length=0.5, method="biweight") -> np.ndarray:
    """
    Detrend the light curve using wotan.flatten and re-normalize by the median.

    A window_length of 0.5 days is chosen to suppress stellar rotation/variability signals
    on timescales longer than the expected transit duration (~3.3 hours) while preserving
    the transit shape.
    """
    flat_flux = wotan.flatten(
        time,
        flux,
        window_length=window_length,
        method=method
    )
    flat_flux = flat_flux / np.median(flat_flux)
    return flat_flux

def run_tls(time, flux, period_min=0.5, period_max=15.0, show_progress_bar=False):
    """
    Runs Transit Least Squares (TLS) search with explicit period bounds.
    """
    model = transitleastsquares(time, flux)
    results = model.power(
        period_min=period_min,
        period_max=period_max,
        show_progress_bar=show_progress_bar
    )
    return results
