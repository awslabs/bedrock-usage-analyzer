# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Path resolution for metadata files using platformdirs."""

import os
from pathlib import Path
from typing import List, Optional

# The package requires Python >= 3.9, where importlib.resources.files exists
from importlib.resources import files  # nosemgrep: python.lang.compatibility.python37.python37-compatibility-importlib2

from platformdirs import user_data_dir

APP_NAME = "bedrock-usage-analyzer"
ENV_VAR = "BEDROCK_ANALYZER_DATA_DIR"

# Set by --update-bundle: the checkout's metadata directory stands in for both the user
# directory and the bundled one, so a maintainer run reads and writes the checkout only
_checkout_metadata: Optional[Path] = None


def use_checkout_metadata(path: Optional[Path]) -> None:
    """Resolve every metadata read and write to ``path`` (None: back to normal)."""
    global _checkout_metadata
    _checkout_metadata = path


def get_user_data_dir() -> Path:
    """Get writable user data directory (env var or platformdirs)."""
    if _checkout_metadata is not None:
        return _checkout_metadata
    if custom := os.environ.get(ENV_VAR):
        return Path(custom).expanduser()
    return Path(user_data_dir(APP_NAME))


def get_bundled_data_dir() -> Path:
    """Get bundled metadata directory (read-only)."""
    if _checkout_metadata is not None:
        return _checkout_metadata
    return files("bedrock_usage_analyzer.metadata")


def get_bundled_file(filename: str) -> Optional[str]:
    """Filesystem path of a bundled metadata file, or None.

    Only returns a path that stays valid after this call, i.e. when the package
    is installed as plain files. For zip/egg installs use load_bundled_yaml().
    """
    try:
        resource = get_bundled_data_dir() / filename
        if isinstance(resource, Path) and resource.is_file():
            return str(resource)
    except (TypeError, FileNotFoundError, ModuleNotFoundError):
        pass
    return None


def load_bundled_yaml(filename: str):
    """Parse a bundled YAML file straight from package resources (works for zip installs too)."""
    import yaml
    try:
        resource = get_bundled_data_dir() / filename
        if resource.is_file():
            return yaml.safe_load(resource.read_text(encoding='utf-8'))
    except (TypeError, FileNotFoundError, ModuleNotFoundError, OSError):
        pass
    return None


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
    bundled = get_bundled_file(filename)
    if bundled:
        return bundled

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


def list_data_files(pattern: str = "*.yml") -> List[Path]:
    """List metadata files matching pattern, one per file name.

    The user's copy of a file wins; bundled files fill in the rest. Refreshing a
    single region therefore does not hide every other bundled region.
    """
    found = {}
    try:
        bundled = get_bundled_data_dir()
        # Only real directories give paths that stay valid after this call (as_file would
        # extract a zipped package to a temporary directory and delete it on exit)
        if isinstance(bundled, Path):
            for path in bundled.glob(pattern):
                found[path.name] = path
    except (TypeError, FileNotFoundError, ModuleNotFoundError):
        pass
    user_dir = get_user_data_dir()
    if user_dir.exists():
        for path in user_dir.glob(pattern):
            found[path.name] = path
    return [found[name] for name in sorted(found)]


def list_data_names(pattern: str = "*.yml") -> List[str]:
    """Names of the metadata files matching pattern (user copies and bundled ones).

    Unlike list_data_files this also sees bundled files of a zipped package, which have no
    file path; read them with yaml_handler.load_data_file.
    """
    import fnmatch
    user_dir = get_user_data_dir()
    names = {p.name for p in user_dir.glob(pattern)} if user_dir.exists() else set()
    try:  # the bundled directory once, also inside a zipped package
        names.update(r.name for r in get_bundled_data_dir().iterdir() if fnmatch.fnmatch(r.name, pattern))
    except (TypeError, FileNotFoundError, ModuleNotFoundError, OSError):
        pass
    return sorted(names)


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
    if _checkout_metadata is not None:
        return f"Metadata will be saved to the checkout: {_checkout_metadata} (--update-bundle; user copies are left alone)"
    user_dir = get_user_data_dir()
    env_set = os.environ.get(ENV_VAR)
    
    if env_set:
        return f"Metadata will be saved to: {user_dir} ({ENV_VAR})"
    else:
        return f"Metadata will be saved to: {user_dir}\n  Tip: Set {ENV_VAR} to use a different location"


def get_default_results_dir() -> Path:
    """Get default results directory (user data dir / results)."""
    return get_user_data_dir() / "results"
