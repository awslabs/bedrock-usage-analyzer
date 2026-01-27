# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Foundation model list management"""

import os
import logging
from typing import List, Dict

from bedrock_usage_analyzer.utils.yaml_handler import load_yaml, save_yaml
from bedrock_usage_analyzer.utils.paths import get_writable_path, get_bundle_path, get_data_path
from bedrock_usage_analyzer.aws.bedrock import (
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


def refresh_region(region, update_bundle: bool = False):
    """Refresh foundation models for a region
    
    Also refreshes prefix mapping, merging with existing prefixes.
    
    Args:
        region: AWS region name (string) or region object (dict)
        update_bundle: Also update bundled metadata (for maintainers)
    """
    # Handle both old format (strings) and new format (dicts)
    if isinstance(region, dict):
        region_name = region['name']
        region_display = region.get('display_name', region_name)
    else:
        region_name = region
        region_display = region_name
    
    logger.info(f"\nProcessing region: {region_display} ({region_name})")
    
    # Refresh prefix mapping - merge with existing
    logger.info("  Refreshing prefix mapping...")
    discovered = discover_prefix_mapping(region_name)
    
    # Load existing prefixes if file exists
    existing_prefixes = {}
    prefix_file = get_writable_path('prefix-mapping.yml')
    try:
        existing_data = load_yaml(str(prefix_file))
        existing_prefixes = {p['prefix']: p for p in existing_data.get('prefixes', [])}
    except (FileNotFoundError, Exception):
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
    prefix_data = {'prefixes': all_prefixes}
    
    save_yaml(str(prefix_file), prefix_data)
    logger.info(f"  ✓ Prefix mapping saved: {prefix_file}")
    
    if update_bundle:
        bundle_path = get_bundle_path()
        if bundle_path:
            bundle_prefix_file = bundle_path / 'prefix-mapping.yml'
            save_yaml(str(bundle_prefix_file), prefix_data)
            logger.info(f"  ✓ Prefix mapping saved: {bundle_prefix_file} (bundled)")
    
    logger.info(f"  ({len(discovered)} discovered, {len(all_prefixes)} total prefixes)")
    
    output_file = get_writable_path(f'fm-list-{region_name}.yml')
    
    # Fetch foundation models
    models = fetch_foundation_models(region_name)
    if models is None:
        return
    
    # Load existing models to preserve quota mappings
    existing_models = load_existing_models(output_file)
    
    # Fetch ALL inference profiles once
    logger.info(f"  Fetching inference profiles...")
    all_profiles = fetch_all_inference_profiles(region_name)
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
    models_data = {'models': sorted(updated_models, key=lambda x: (x['provider'], x['model_id']))}
    save_yaml(str(output_file), models_data)
    logger.info(f"  ✓ Saved {len(updated_models)} models to {output_file}")
    
    if update_bundle:
        bundle_path = get_bundle_path()
        if bundle_path:
            bundle_file = bundle_path / f'fm-list-{region}.yml'
            save_yaml(str(bundle_file), models_data)
            logger.info(f"  ✓ Saved: {bundle_file} (bundled)")


def refresh_all_regions(regions: List, update_bundle: bool = False):
    """Refresh foundation models for all regions
    
    Args:
        regions: List of AWS region names (strings) or region objects (dicts)
        update_bundle: Also update bundled metadata (for maintainers)
    """
    for region in regions:
        # Handle both old format (strings) and new format (dicts)
        if isinstance(region, dict):
            region_name = region['name']
        else:
            region_name = region
        
        refresh_region(region_name, update_bundle=update_bundle)
