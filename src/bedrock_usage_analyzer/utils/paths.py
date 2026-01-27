# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Path resolution for metadata files using platformdirs."""

import os
from pathlib import Path
from typing import Union, Optional

try:
    from importlib.resources import files, as_file
except ImportError:
    from importlib_resources import files, as_file

from platformdirs import user_data_dir

APP_NAME = "bedrock-usage-analyzer"
ENV_VAR = "BEDROCK_ANALYZER_DATA_DIR"


def get_user_data_dir() -> Path:
    """Get writable user data directory (env var or platformdirs)."""
    if custom := os.environ.get(ENV_VAR):
        return Path(custom).expanduser()
    return Path(user_data_dir(APP_NAME))


def get_bundled_data_dir() -> Path:
    """Get bundled metadata directory (read-only)."""
    return files("bedrock_usage_analyzer.metadata")


def get_data_path(filename: str) -> str:
    """Get path for reading a metadata file.
    
    Priority: env var → platformdirs → bundled
    """
    # Check user data dir first
    user_dir = get_user_data_dir()
    user_file = user_dir / filename
    if user_file.exists():
        return str(user_file)
    
    # Fall back to bundled
    try:
        bundled = get_bundled_data_dir()
        with as_file(bundled / filename) as path:
            if path.exists():
                return str(path)
    except (TypeError, FileNotFoundError, ModuleNotFoundError):
        pass
    
    # Return user path even if doesn't exist (for error messages)
    return str(user_file)


def get_writable_path(filename: str) -> Path:
    """Get path for writing a metadata file.
    
    Always returns user data dir (env var or platformdirs).
    Creates directory if needed.
    """
    user_dir = get_user_data_dir()
    user_dir.mkdir(parents=True, exist_ok=True)
    return user_dir / filename


def get_bundle_path() -> Optional[Path]:
    """Get bundled data path if in dev environment.
    
    Returns None if not in a cloned repo (for --update-bundle flag).
    """
    bundle = Path("./src/bedrock_usage_analyzer/metadata")
    if bundle.is_dir():
        return bundle
    return None


def list_data_files(pattern: str = "*.yml") -> list[Path]:
    """List metadata files matching pattern.
    
    Returns files from user data dir if exists, else bundled.
    """
    user_dir = get_user_data_dir()
    if user_dir.exists():
        files_list = list(user_dir.glob(pattern))
        if files_list:
            return files_list
    
    # Fall back to bundled
    try:
        bundled = get_bundled_data_dir()
        with as_file(bundled) as bundled_path:
            return list(bundled_path.glob(pattern))
    except (TypeError, FileNotFoundError, ModuleNotFoundError):
        return []


def is_using_customized_metadata() -> bool:
    """Check if using customized (user-refreshed) metadata."""
    user_dir = get_user_data_dir()
    if not user_dir.exists():
        return False
    return (user_dir / "regions.yml").exists() or any(user_dir.glob("fm-list-*.yml"))


def get_metadata_location_message() -> str:
    """Get user-friendly message about metadata location for analysis."""
    user_dir = get_user_data_dir()
    env_set = os.environ.get(ENV_VAR)
    customized = is_using_customized_metadata()
    
    if customized:
        if env_set:
            return f"Using customized metadata from: {user_dir} ({ENV_VAR})"
        else:
            return f"Using customized metadata from: {user_dir}"
    else:
        msg = "Using default metadata (bundled with package)"
        if env_set:
            msg += f"\n  Note: {ENV_VAR} is set but directory is empty"
        else:
            msg += f"\n  Tip: Run 'bua refresh' commands to customize"
        return msg


def get_refresh_location_message() -> str:
    """Get user-friendly message about where refresh will save."""
    user_dir = get_user_data_dir()
    env_set = os.environ.get(ENV_VAR)
    
    if env_set:
        return f"Metadata will be saved to: {user_dir} ({ENV_VAR})"
    else:
        return f"Metadata will be saved to: {user_dir}\n  Tip: Set {ENV_VAR} to use a different location"


def get_default_results_dir() -> Path:
    """Get default results directory (user data dir / results)."""
    return get_user_data_dir() / "results"
