import lightkurve as lk

# Search for a known TESS target
search_result = lk.search_lightcurve(
    "TIC 25155310",
    mission="TESS",
    cadence="short"
)

print(search_result)

# Download the first available light curve
lc = search_result.download()

print("\nDownload successful!")
print(lc)