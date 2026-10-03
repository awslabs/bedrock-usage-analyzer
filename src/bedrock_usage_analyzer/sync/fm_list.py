# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Foundation model list management"""

import logging
from typing import List, Dict

from bedrock_usage_analyzer.utils.yaml_handler import load_fm_list, load_yaml, model_endpoints, save_yaml, valid_models
from bedrock_usage_analyzer.utils.paths import get_writable_path, get_bundle_path
from bedrock_usage_analyzer.aws.bedrock import (
    fetch_foundation_models,
    list_system_profiles,
    build_profile_map,
    discover_prefix_mapping,
    load_prefix_mapping,
    prefix_mapping_layers,
    QUOTA_KEYWORD_ON_DEMAND,
    QUOTA_KEYWORD_GLOBAL
)

logger = logging.getLogger(__name__)


def load_existing_models(region: str) -> Dict[str, Dict]:
    """Saved models of the region by ID (user copy, else bundled); {} when there is none.

    Malformed entries are skipped (load_fm_list), so one bad entry cannot drop every saved
    quota mapping of the region. A file path (the argument in earlier releases) is still
    read as that file.
    """
    try:
        if str(region).endswith(('.yml', '.yaml')):
            return {m['model_id']: m for m in valid_models(load_yaml(str(region)))}
        return {m['model_id']: m for m in load_fm_list(region) or []}
    except Exception as e:
        logger.warning(f"Could not load existing models for {region}: {e}")
        return {}


def save_models(filepath: str, models: List[Dict]):
    """Save models to YAML file
    
    Args:
        filepath: Path to YAML file
        models: List of model dictionaries
    """
    sorted_models = sorted(models, key=lambda x: (x['provider'], x['model_id']))
    save_yaml(filepath, {'models': sorted_models})


def _empty_endpoint() -> Dict:
    """A new endpoint entry: every quota slot, unmapped (fm-quotas fills them in)."""
    return {'quotas': {'concurrent': None, 'rpm': None, 'tpd': None, 'tpm': None}}


def refresh_region(region_name: str, update_bundle: bool = False):
    """Refresh foundation models for a region

    Also refreshes prefix mapping, merging with existing prefixes.

    Args:
        region_name: AWS region name
        update_bundle: Also update bundled metadata (for maintainers)
    """
    logger.info(f"\nProcessing region: {region_name}")

    # Models first: a region the account cannot call is skipped before anything else is
    # listed or written
    models = fetch_foundation_models(region_name)
    if models is None:
        return

    # List the system inference profiles once; both the prefix discovery and the
    # model -> profile map below are built from it
    logger.info("  Fetching inference profiles...")
    all_profiles = list_system_profiles(region_name)
    profiles_listed = all_profiles is not None  # False: the listing failed (not "none listed")
    all_profiles = all_profiles or []

    # Refresh prefix mapping - merge with existing
    logger.info("  Refreshing prefix mapping...")
    discovered = discover_prefix_mapping(region_name, all_profiles)

    # Known = bundled + user entries (keeps prefixes of other partitions, e.g. us-gov).
    # The user file only ever gets the user's own and newly discovered entries: a full copy
    # of the bundle there would hide later bundled changes to existing prefixes.
    prefix_file = get_writable_path('prefix-mapping.yml')
    bundled, user = prefix_mapping_layers()
    user_entries = {p['prefix']: p for p in user}
    known = {**{p['prefix']: p for p in bundled}, **user_entries}
    new_entries = [e for e in discovered if e['prefix'] not in known]

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
    
    def by_prefix(entries):
        return sorted(entries.values(), key=lambda x: x['prefix'])

    if new_entries:  # an unchanged user file is not rewritten
        for entry in new_entries:
            user_entries[entry['prefix']] = entry
        save_yaml(str(prefix_file), {'prefixes': by_prefix(user_entries)})
        logger.info(f"  ✓ Prefix mapping saved: {prefix_file}")

    all_prefixes = dict(known)
    for entry in manual_entries + new_entries:
        all_prefixes[entry['prefix']] = entry

    if update_bundle:
        bundle_path = get_bundle_path()
        if bundle_path:
            bundle_prefix_file = bundle_path / 'prefix-mapping.yml'
            # The file being rewritten (the checkout), not the installed package's copy
            bundled = {p['prefix']: p for p in
                       ((load_yaml(str(bundle_prefix_file)) if bundle_prefix_file.exists() else None) or {}).get('prefixes') or []}
            # Everything discovered that the bundle lacks, including prefixes already saved in
            # the user's own file by an earlier refresh without --update-bundle
            for entry in manual_entries + [e for e in discovered if e['prefix'] not in bundled]:
                bundled[entry['prefix']] = entry
            save_yaml(str(bundle_prefix_file), {'prefixes': by_prefix(bundled)})
            logger.info(f"  ✓ Prefix mapping saved: {bundle_prefix_file} (bundled)")

    # Later lookups in this run must see newly discovered prefixes
    load_prefix_mapping(refresh=True)
    logger.info(f"  ({len(discovered)} discovered, {len(new_entries)} new, {len(all_prefixes)} total prefixes)")
    
    output_file = get_writable_path(f'fm-list-{region_name}.yml')

    # Load existing models to preserve quota mappings
    # User copy if present, else the bundled list, so refreshing never drops quota mappings
    existing_models = load_existing_models(region_name)
    
    # Build mapping from model to inference profiles
    profile_map = build_profile_map(all_profiles)
    logger.info(f"  Found {len(profile_map)} models with inference profiles")
    
    # Update models with profile information
    updated_models = []
    for model in models:
        model_id = model['model_id']
        
        # Preserve existing endpoints/quotas if they exist
        if model_id in existing_models:
            model['endpoints'] = dict(model_endpoints(existing_models[model_id]))
            if not profiles_listed and existing_models[model_id].get('inference_profiles'):
                # The listing failed: the saved profiles stay with the endpoints they belong to
                model['inference_profiles'] = existing_models[model_id]['inference_profiles']
        # The endpoints follow the model: 'base' only while it is invokable on demand, and a
        # profile endpoint only while the region lists that profile (when the listing worked,
        # so a failed listing never drops saved quota mappings)
        endpoints = model.setdefault('endpoints', {})
        if 'ON_DEMAND' not in model.get('inference_types', []):
            endpoints.pop('base', None)
        if profiles_listed:
            listed = set(profile_map.get(model_id, []))
            for prefix in [p for p in endpoints if p != 'base' and p not in listed]:
                del endpoints[prefix]
        if 'ON_DEMAND' in model.get('inference_types', []):
            endpoints.setdefault('base', _empty_endpoint())
        
        # Add inference profiles if available
        if model_id in profile_map:
            model['inference_profiles'] = profile_map[model_id]
            
            # Initialize endpoint structures for each profile prefix
            for prefix in profile_map[model_id]:
                endpoints.setdefault(prefix, _empty_endpoint())
        
        updated_models.append(model)
    
    # Save updated models
    models_data = {'models': sorted(updated_models, key=lambda x: (x['provider'], x['model_id']))}
    save_yaml(str(output_file), models_data)
    logger.info(f"  ✓ Saved {len(updated_models)} models to {output_file}")
    
    if update_bundle:
        bundle_path = get_bundle_path()
        if bundle_path:
            bundle_file = bundle_path / f'fm-list-{region_name}.yml'
            save_yaml(str(bundle_file), models_data)
            logger.info(f"  ✓ Saved: {bundle_file} (bundled)")


def refresh_all_regions(regions: List[str], update_bundle: bool = False):
    """Refresh foundation models for all regions

    Args:
        regions: List of AWS region names
        update_bundle: Also update bundled metadata (for maintainers)
    """
    for region in regions:
        refresh_region(region, update_bundle=update_bundle)
