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


def load_fm_list(region):
    """Models of a region's fm-list (user copy, else bundled); None when there is no list.

    A file that is empty, has no 'models' key, or 'models: null' gives [].
    """
    import os
    from bedrock_usage_analyzer.utils.paths import get_data_path
    path = get_data_path(f'fm-list-{region}.yml')
    if not os.path.exists(path):
        return None
    data = load_yaml(path)
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
