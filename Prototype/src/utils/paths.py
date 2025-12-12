"""
Simple helper to manage all important folder paths in one place.

This way we are not hardcoding 'results/', 'outputs/' etc. everywhere.
"""

import os
from dataclasses import dataclass


@dataclass
class ProjectPaths:
    data_raw: str
    data_processed: str
    figures_dir: str
    results_dir: str
    outputs_dir: str
    checkpoints_dir: str
    predictions_dir: str

    def make_dirs(self):
        """Create all required directories if they don't exist."""
        for p in [
            self.data_raw,
            self.data_processed,
            self.figures_dir,
            self.results_dir,
            self.outputs_dir,
            self.checkpoints_dir,
            self.predictions_dir,
        ]:
            os.makedirs(p, exist_ok=True)