# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Foundation model list management"""

import os
import logging
from typing import List, Dict

from bedrock_analyzer.utils.yaml_handler import load_yaml, save_yaml
from bedrock_analyzer.aws.bedrock import (
    fetch_foundation_models,
    fetch_all_inference_profiles,
    build_profile_map,
    discover_prefix_mapping,
    QUOTA_KEYWORD_ON_DEMAND,
    QUOTA_KEYWORD_GLOBAL
)

logger = logging.getLogger(__name__)


def load_existing_models(filepath: str) -> Dict[str, Dict]:
    """Load existing models from YAML file
    
    Args:
        filepath: Path to YAML file
        
    Returns:
        Dictionary mapping model IDs to model data
    """
    try:
        data = load_yaml(filepath)
        if data and 'models' in data:
            return {m['model_id']: m for m in data['models']}
    except (FileNotFoundError, Exception) as e:
        logger.warning(f"Could not load existing models from {filepath}: {e}")
    return {}


def save_models(filepath: str, models: List[Dict]):
    """Save models to YAML file
    
    Args:
        filepath: Path to YAML file
        models: List of model dictionaries
    """
    sorted_models = sorted(models, key=lambda x: (x['provider'], x['model_id']))
    save_yaml(filepath, {'models': sorted_models})


def refresh_region(region: str):
    """Refresh foundation models for a region
    
    Also refreshes prefix mapping, merging with existing prefixes.
    
    Args:
        region: AWS region name
    """
    logger.info(f"\nProcessing region: {region}")
    
    # Refresh prefix mapping - merge with existing
    logger.info("  Refreshing prefix mapping...")
    discovered = discover_prefix_mapping(region)
    
    # Load existing prefixes if file exists
    existing_prefixes = {}
    prefix_file = 'metadata/prefix-mapping.yml'
    if os.path.exists(prefix_file):
        try:
            existing_data = load_yaml(prefix_file)
            existing_prefixes = {p['prefix']: p for p in existing_data.get('prefixes', [])}
        except Exception:
            pass
    
    # Manual entries (always include)
    manual_entries = [
        {
            'prefix': 'base',
            'quota_keyword': QUOTA_KEYWORD_ON_DEMAND,
            'description': 'on-demand',
            'is_regional': False,
            'source': 'manual'
        },
        {
            'prefix': 'global',
            'quota_keyword': QUOTA_KEYWORD_GLOBAL,
            'description': 'global inference profile',
            'is_regional': False,
            'source': 'manual'
        }
    ]
    
    # Merge: manual entries + existing + newly discovered
    for entry in manual_entries:
        existing_prefixes[entry['prefix']] = entry
    
    for entry in discovered:
        if entry['prefix'] not in existing_prefixes:
            existing_prefixes[entry['prefix']] = entry
    
    # Sort by prefix for consistency
    all_prefixes = sorted(existing_prefixes.values(), key=lambda x: x['prefix'])
    
    save_yaml(prefix_file, {'prefixes': all_prefixes})
    logger.info(f"  ✓ Prefix mapping refreshed ({len(discovered)} discovered, {len(all_prefixes)} total)")
    
    output_file = f'metadata/fm-list-{region}.yml'
    
    # Fetch foundation models
    models = fetch_foundation_models(region)
    if models is None:
        return
    
    # Load existing models to preserve quota mappings
    existing_models = load_existing_models(output_file)
    
    # Fetch ALL inference profiles once
    logger.info(f"  Fetching inference profiles...")
    all_profiles = fetch_all_inference_profiles(region)
    # Build mapping from model to inference profiles
    profile_map = build_profile_map(all_profiles)
    logger.info(f"  Found {len(profile_map)} models with inference profiles")
    
    # Update models with profile information
    updated_models = []
    for model in models:
        model_id = model['model_id']
        
        # Preserve existing endpoints/quotas if they exist
        if model_id in existing_models:
            existing = existing_models[model_id]
            model['endpoints'] = existing.get('endpoints', {})
        else:
            # Initialize endpoints structure for ON_DEMAND models
            if 'ON_DEMAND' in model.get('inference_types', []):
                model['endpoints'] = {
                    'base': {
                        'quotas': {
                            'concurrent': None,
                            'rpm': None,
                            'tpd': None,
                            'tpm': None
                        }
                    }
                }
        
        # Add inference profiles if available
        if model_id in profile_map:
            model['inference_profiles'] = profile_map[model_id]
            
            # Initialize endpoint structures for each profile prefix
            if 'endpoints' not in model:
                model['endpoints'] = {}
            
            for prefix in profile_map[model_id]:
                if prefix not in model['endpoints']:
                    model['endpoints'][prefix] = {
                        'quotas': {
                            'concurrent': None,
                            'rpm': None,
                            'tpd': None,
                            'tpm': None
                        }
                    }
        
        updated_models.append(model)
    
    # Save updated models
    save_models(output_file, updated_models)
    logger.info(f"  ✓ Saved {len(updated_models)} models to {output_file}")


def refresh_all_regions(regions: List[str]):
    """Refresh foundation models for all regions
    
    Args:
        regions: List of AWS region names
    """
    for region in regions:
        refresh_region(region)
