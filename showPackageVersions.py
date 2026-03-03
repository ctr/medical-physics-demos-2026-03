#!/usr/bin/env python3

# Show the versions of pypulseq and MRzeroCore installed in current venv for debugging.

from importlib.metadata import version, PackageNotFoundError, distribution

for pkg in ["pypulseq", "MRzeroCore", "sigpy"]:
    try:
        dist = distribution(pkg)
        print(f"{pkg}:")
        print("  Version:", dist.version)
        print("  Location:", dist.locate_file(""))  # folder of the package
        print("  Installer:", dist.metadata.get("Installer", "unknown"))
    except PackageNotFoundError:
        print(f"{pkg} is not installed")


