#!/usr/bin/env python3
"""
Show the versions of selected packages installed in the current virtual environment.
Can be imported and used via showPackageVersions()
or run directly as a script.
"""

from importlib.metadata import version, PackageNotFoundError, distribution


def showPackageVersions(packages=None):
    """
    Print version, location, and installer information
    for the given list of package names.

    Parameters
    ----------
    packages : list[str] | None
        List of package names. If None, defaults to
        ["pypulseq", "MRzeroCore", "sigpy"].
    """
    if packages is None:
        packages = ["pypulseq", "MRzeroCore", "sigpy"]

    for pkg in packages:
        try:
            dist = distribution(pkg)
            print(f"{pkg}:")
            print("  Version:", dist.version)
            print("  Location:", dist.locate_file(""))  # folder of the package
            print("  Installer:", dist.metadata.get("Installer", "unknown"))
        except PackageNotFoundError:
            print(f"{pkg} is not installed")


if __name__ == "__main__":
    showPackageVersions()
