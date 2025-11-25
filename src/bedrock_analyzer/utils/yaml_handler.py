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


def save_yaml(filepath, data):
    """Save data to YAML file with UTF-8 encoding
    
    Args:
        filepath: Path to YAML file
        data: Data to save
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
