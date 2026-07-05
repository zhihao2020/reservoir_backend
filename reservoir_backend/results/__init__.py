"""Result manifest, catalog, and export helpers.

This package is the M7 result-contract layer. It does not run solvers, mutate
existing reports, or implement any frontend / UDP protocol.
"""

from .catalog import ResultCatalog
from .manifest import ResultManifest, validate_result_manifest
from .report_index import build_report_path_index

__all__ = [
    "ResultCatalog",
    "ResultManifest",
    "build_report_path_index",
    "validate_result_manifest",
]
