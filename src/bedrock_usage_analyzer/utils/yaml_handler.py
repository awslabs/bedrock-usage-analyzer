# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Centralized YAML file operations with UTF-8 encoding"""

import yaml


def load_yaml(filepath):
    """Load YAML file with UTF-8 encoding
    
    Args:
        filepath: Path to YAML file
        
    Returns:
        dict: Parsed YAML data
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_data_file(filename):
    """Parsed metadata file: the user copy, else the bundled one (also from a zipped
    package); None when neither exists."""
    import os
    from bedrock_usage_analyzer.utils.paths import get_data_path, load_bundled_yaml
    path = get_data_path(filename)
    # An existing but empty user file is {} (present), not None (missing)
    return (load_yaml(path) or {}) if os.path.exists(path) else load_bundled_yaml(filename)


def load_fm_list(region):
    """Models of a region's fm-list (user copy, else bundled); None when there is no list.

    A file that is empty, has no 'models' key, or 'models: null' gives [].
    """
    data = load_data_file(f'fm-list-{region}.yml')
    return None if data is None else valid_models(data)


def fm_file_data(data):
    """Parsed fm-list ``data`` as a dict whose 'models' is a list, ready to be written back.

    Malformed entries and other top-level keys are kept; valid_models() of the result
    returns the same entry dicts, so changes to them reach the file.
    """
    data = data if isinstance(data, dict) else {}
    if not isinstance(data.get('models'), list):
        data['models'] = []
    return data


def valid_models(data):
    """The usable model entries of parsed fm-list ``data``: 'models: null', a non-mapping file
    and entries without a model_id are skipped. The one rule every reader applies."""
    models = data.get('models') if isinstance(data, dict) else None
    return [m for m in models or [] if isinstance(m, dict) and m.get('model_id')]


def save_yaml(filepath, data):
    """Save data to YAML file with UTF-8 encoding
    
    Args:
        filepath: Path to YAML file
        data: Data to save
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


def fm_endpoints(models, model_id):
    """Endpoint keys ('base', 'us', 'global', ...) of ``model_id`` in parsed fm-list models.

    None when the model is not listed. The one rule for "does the list have this endpoint".
    """
    for model in models or []:
        if model.get('model_id') == model_id:
            return set(model.get('endpoints') or {})
    return None


def has_endpoint(models, model_id, prefix):
    """True when the fm-list has ``model_id`` with the endpoint of ``prefix`` (None: base)."""
    return (prefix or 'base') in (fm_endpoints(models, model_id) or set())


def quota_slots(models):
    """(model ID, endpoint, metric, code) of every mapped quota in parsed fm-list models.

    Null endpoints ('us: null'), null quotas and entries without a code are skipped.
    """
    for model in models or []:
        for endpoint, endpoint_data in (model.get('endpoints') or {}).items():
            if not isinstance(endpoint_data, dict):
                continue
            for metric, quota in (endpoint_data.get('quotas') or {}).items():
                if isinstance(quota, dict) and quota.get('code'):
                    yield (model['model_id'], endpoint, metric, quota['code'])
