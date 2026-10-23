"""
Checksums and library versions recorded with every run.

Small, stateless helpers shared by the run scripts, so that each script records the same
things in the same way. This module contains no execution code.
"""

import hashlib
import platform

import numpy as np
import pandas as pd
import sklearn


def md5_of_file(path: str) -> str:
    """MD5 of a file's bytes, read in chunks so a large file never has to fit in memory."""
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5_of_frame(df: pd.DataFrame) -> str:
    """MD5 of a table as it would be written to CSV (without the index)."""
    return hashlib.md5(df.to_csv(index=False).encode()).hexdigest()


def library_versions() -> dict:
    """Python and the three libraries whose version can change a result."""
    return {"python": platform.python_version(), "pandas": pd.__version__,
            "numpy": np.__version__, "scikit-learn": sklearn.__version__}
