import lightkurve as lk

search = lk.search_lightcurve(
    "TIC 25155310",
    mission="TESS",
    author="SPOC",
    sector=1
)

lc = search.download()

print("\nDownloaded successfully!\n")

# Print the exact FITS file path
print("FITS file location:")
print(lc.meta["FILENAME"])