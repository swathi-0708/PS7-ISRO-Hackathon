import lightkurve as lk

search = lk.search_lightcurve(
    "TIC 25155310",
    mission="TESS",
    cadence="short"
)

lc = search.download()

print(lc)

print("\nColumns:\n")
print(lc.columns)

print("\nMetadata:\n")
print(lc.meta)