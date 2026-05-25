"""
This file should:

load YAML configs
validate config existence
return dictionary object

Then:

run_pipeline.py
    ↓
config_loader
    ↓
StreamLoader
    ↓
VideoCaptureManager
"""

from pathlib import Path
import yaml


class ConfigLoader:
    """
    Utility class for loading YAML configuration files.
    """

    @staticmethod
    def load_yaml(config_path: str) -> dict:
        """
        Load YAML configuration file.

        Parameters
        ----------
        config_path : str
            Path to YAML configuration file.

        Returns
        -------
        dict
            Parsed YAML configuration.
        """

        config_file = Path(config_path)

        if not config_file.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}"
            )

        with open(config_file, "r") as file:
            config = yaml.safe_load(file)

        return config