# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""AWS Bedrock service operations"""

import sys
import logging
from typing import List, Dict, Optional, Tuple

from bedrock_usage_analyzer.aws.client_factory import create_client
from bedrock_usage_analyzer.utils.partition import build_arn

logger = logging.getLogger(__name__)


# Quota keyword constants
QUOTA_KEYWORD_ON_DEMAND = 'on-demand'
QUOTA_KEYWORD_CROSS_REGION = 'cross-region'
QUOTA_KEYWORD_GLOBAL = 'global'

# Used only when no prefix-mapping.yml can be read at all (the bundled file is the source
# of truth and normally always present)
FALLBACK_PROFILE_PREFIXES = frozenset({'us', 'eu', 'apac', 'jp', 'au', 'ca', 'in', 'us-gov', 'global'})

# Cache for prefix mapping to avoid repeated file reads
_prefix_mapping_cache = None


def _read_prefixes(path) -> List[Dict]:
    from bedrock_usage_analyzer.utils.yaml_handler import load_yaml
    try:
        return (load_yaml(str(path)) or {}).get('prefixes', []) or []
    except (FileNotFoundError, OSError):
        return []


def _load_prefix_mapping() -> List[Dict]:
    """Load the prefix mapping: bundled entries, overridden by the user's copy

    Merging (instead of letting the user file hide the bundled one) keeps
    prefixes added in newer releases, such as 'us-gov' and 'in', available to
    users whose prefix-mapping.yml was written by an older version.

    Returns:
        List of prefix mapping dictionaries

    Raises:
        FileNotFoundError: If no prefix-mapping.yml exists at all
    """
    global _prefix_mapping_cache

    if _prefix_mapping_cache is not None:
        return _prefix_mapping_cache

    bundled, user = prefix_mapping_layers()
    merged: Dict[str, Dict] = {}
    for entry in bundled + user:
        merged[entry['prefix']] = entry

    if not merged:
        raise FileNotFoundError(
            "\nprefix-mapping.yml not found!\n"
            "Please run: bua refresh fm-list\n"
            "This will refresh both foundation model lists and prefix mapping."
        )
    _prefix_mapping_cache = sorted(merged.values(), key=lambda m: m['prefix'])
    return _prefix_mapping_cache


def prefix_mapping_layers():
    """(bundled entries, user entries) of prefix-mapping.yml, each read once."""
    from bedrock_usage_analyzer.utils.paths import get_user_data_dir, load_bundled_yaml
    bundled = list((load_bundled_yaml('prefix-mapping.yml') or {}).get('prefixes', []) or [])
    return bundled, list(_read_prefixes(get_user_data_dir() / 'prefix-mapping.yml'))


def load_prefix_mapping(refresh: bool = False) -> List[Dict]:
    """Public accessor for the merged prefix mapping; ``refresh`` re-reads the files."""
    global _prefix_mapping_cache
    if refresh:
        _prefix_mapping_cache = None
    try:
        return list(_load_prefix_mapping())
    except FileNotFoundError:
        return []


def get_endpoint_quota_keywords() -> Dict[str, str]:
    """Get mapping of endpoint prefix to quota keyword
    
    Returns:
        Dict mapping prefix to quota keyword (e.g., {'base': 'on-demand', 'us': 'cross-region'})
    """
    mapping = _load_prefix_mapping()
    return {m['prefix']: m['quota_keyword'] for m in mapping}


def get_endpoint_descriptions() -> Dict[str, str]:
    """Get mapping of endpoint prefix to description
    
    Returns:
        Dict mapping prefix to description (e.g., {'base': 'on-demand', 'us': 'cross-region inference profile'})
    """
    mapping = _load_prefix_mapping()
    return {m['prefix']: m['description'] for m in mapping}


def get_regional_profile_prefixes() -> List[str]:
    """Get list of regional profile prefixes
    
    Returns:
        List of regional prefixes (e.g., ['us', 'eu', 'jp', 'au', 'apac', 'ca'])
    """
    try:
        mapping = _load_prefix_mapping()
    except FileNotFoundError:
        mapping = []
    if not mapping:
        # No prefix-mapping.yml at all: fall back to the prefixes this release knows
        return sorted(FALLBACK_PROFILE_PREFIXES - {'global'})
    # prefix-mapping.yml is the single source of truth once it exists
    return sorted(m['prefix'] for m in mapping if m.get('is_regional'))


def get_default_region_prefix_map() -> Dict[str, str]:
    """Get mapping of region prefix to system profile prefix
    
    Returns:
        Dict mapping region prefix to system profile prefix (e.g., {'us': 'us', 'ap': 'apac'})
    """
    mapping = _load_prefix_mapping()
    result = {m['prefix']: m['prefix'] for m in mapping if m['is_regional']}
    result['ap'] = 'apac'  # Special case: 'ap' region prefix maps to 'apac' system profile
    return result


def get_profile_prefixes() -> frozenset:
    """All system inference profile prefixes, from prefix-mapping.yml (fallback set without it)."""
    try:
        mapped = {m['prefix'] for m in _load_prefix_mapping() if m.get('prefix') != 'base'}
    except FileNotFoundError:
        mapped = set()
    return frozenset(mapped) if mapped else FALLBACK_PROFILE_PREFIXES


def split_profile_id(endpoint_id: str) -> Tuple[str, Optional[str]]:
    """Split an endpoint ID into (model_id, prefix).

    'us.amazon.nova-pro-v1:0' -> ('amazon.nova-pro-v1:0', 'us')
    'deepseek.v3.2'           -> ('deepseek.v3.2', None)   # base model, not a prefix
    """
    if '.' in endpoint_id:
        first, rest = endpoint_id.split('.', 1)
        if first in get_profile_prefixes():
            return rest, first
    return endpoint_id, None


# Second name segments that mark a separate partition rather than a direction
_PARTITION_SEGMENTS = {'gov', 'iso', 'isob', 'isof', 'isoe'}


def region_group(region: str) -> str:
    """Region family used to guess a profile prefix.

    'eu-west-1' -> 'eu', 'us-gov-west-1' -> 'us-gov', 'us-iso-east-1' -> 'us-iso',
    so regions of other partitions never fall into a commercial family.
    """
    parts = region.split('-')
    if len(parts) > 2 and parts[1] in _PARTITION_SEGMENTS:
        return f"{parts[0]}-{parts[1]}"
    return parts[0]


def model_id_from_arn(arn: str) -> Optional[str]:
    """'arn:aws:bedrock:us-east-1::foundation-model/amazon.nova-pro-v1:0' -> 'amazon.nova-pro-v1:0'."""
    if ':foundation-model/' in arn:
        return arn.split(':foundation-model/', 1)[1]
    return None


def region_from_arn(arn: str) -> str:
    """Region field of an ARN ('' for region-less ARNs such as global routing targets)."""
    parts = arn.split(':')
    return parts[3] if len(parts) > 3 else ''


def list_inference_profiles(bedrock_client, type_equals: str) -> List[Dict]:
    """List all inference profiles of one type ('SYSTEM_DEFINED' or 'APPLICATION')."""
    if not hasattr(bedrock_client, 'list_inference_profiles'):
        return []
    profiles = []
    params = {'maxResults': 1000, 'typeEquals': type_equals}
    while True:
        response = bedrock_client.list_inference_profiles(**params)
        profiles.extend(response.get('inferenceProfileSummaries', []))
        token = response.get('nextToken')
        if not token:
            break
        params['nextToken'] = token
    return profiles


def discover_prefix_mapping(region: str, profiles: Optional[List[Dict]] = None) -> List[Dict]:
    """Discover system profile prefixes from Bedrock API
    
    Discovers regional inference profile prefixes (us, eu, jp, au, apac, ca, etc.)
    by analyzing SYSTEM_DEFINED profiles. Automatically classifies as regional
    if model ARNs span multiple regions with same prefix.
    
    Args:
        region: AWS region to use for API calls
        profiles: SYSTEM_DEFINED profiles already listed for the region (skips the API call)

    Returns:
        List of discovered prefix mappings with structure:
        [
            {
                'prefix': 'us',
                'quota_keyword': 'cross-region',
                'description': 'cross-region inference profile',
                'is_regional': True,
                'source': 'discovered'
            },
            ...
        ]
    """
    try:
        all_profiles = profiles if profiles is not None else \
            list_inference_profiles(create_client('bedrock', region), 'SYSTEM_DEFINED')

        # Extract system profile prefixes
        discovered = []
        seen_prefixes = set()
        
        for profile in all_profiles:
            if profile['type'] == 'SYSTEM_DEFINED' and '.' in profile['inferenceProfileId']:
                system_prefix = profile['inferenceProfileId'].split('.')[0]
                
                # Skip if already processed or if it's 'global'
                if system_prefix in seen_prefixes or system_prefix == 'global':
                    continue
                
                model_arns = [m['modelArn'] for m in profile['models']]
                
                # Classify as regional if multiple ARNs in same region prefix
                if len(model_arns) > 1:
                    regions = [region_from_arn(arn) for arn in model_arns]
                    region_prefixes = set(region_group(r) for r in regions)
                    
                    # Regional: all ARNs in same region prefix (us-*, eu-*, etc.)
                    if len(region_prefixes) == 1:
                        discovered.append({
                            'prefix': system_prefix,
                            'quota_keyword': QUOTA_KEYWORD_CROSS_REGION,
                            'description': 'cross-region inference profile',
                            'is_regional': True,
                            'source': 'discovered'
                        })
                        seen_prefixes.add(system_prefix)
        
        logger.info(f"Discovered {len(discovered)} regional prefixes: {[d['prefix'] for d in discovered]}")
        return discovered
        
    except Exception as e:
        logger.warning(f"Failed to discover prefix mapping: {e}")
        return []


def fetch_foundation_models(region: str) -> Optional[List[Dict]]:
    """Fetch foundation models for a region
    
    Args:
        region: AWS region name
        
    Returns:
        List of model dictionaries or None if access denied
    """
    try:
        bedrock = create_client('bedrock', region)
        response = bedrock.list_foundation_models()
        
        models = []
        for model in response.get('modelSummaries', []):
            models.append({
                'model_id': model['modelId'],
                'provider': model['providerName'],
                'inference_types': model.get('inferenceTypesSupported', [])
            })
        
        return models
    
    except Exception as e:
        error_msg = str(e)
        if any(x in error_msg for x in ['AccessDenied', 'UnauthorizedOperation', 'not enabled', 'not subscribed']):
            print(f"  ⊘ Skipping {region} (access denied or not enabled)", file=sys.stderr)
        else:
            print(f"  ✗ Failed to fetch models for {region}: {e}", file=sys.stderr)
        return None


def fetch_all_inference_profiles(region: str) -> List[Dict]:
    """Fetch ALL inference profiles in region
    This fetches only system inference profile, not application inference profile
    The purpose is to list down the available system inference profiles for a given FM.
    
    Args:
        region: AWS region name
        
    Returns:
        List of inference profile dictionaries
    """
    try:
        return list_inference_profiles(create_client('bedrock', region), 'SYSTEM_DEFINED')
    except Exception as e:
        # Inference profiles might not be available in all regions
        logger.warning(f"  Could not list inference profiles in {region}: {e}")
        return []


def build_profile_map(profiles: List[Dict]) -> Dict[str, List[str]]:
    """Build mapping: model_id → [profile_prefixes]
    Basically given a list of inference profiles (each profile with the FM it is for), it builds a map with FM key first, then list of profiles for each FM.
    
    Args:
        profiles: List of inference profile dictionaries
        
    Returns:
        Dictionary mapping model IDs to list of profile prefixes
    """
    profile_map = {}
    
    for profile in profiles:
        profile_id = profile.get('inferenceProfileId', '')
        
        # A system profile ID always starts with its prefix (us, eu, jp, au, apac, global, ...).
        # Taken literally rather than via split_profile_id: new prefixes are discovered here.
        if '.' not in profile_id:
            continue
        prefix = profile_id.split('.')[0]

        # Add this prefix to all models in this profile
        for model in profile.get('models', []):
            model_id = model_id_from_arn(model.get('modelArn', ''))
            if model_id:
                if model_id not in profile_map:
                    profile_map[model_id] = []
                if prefix not in profile_map[model_id]:
                    profile_map[model_id].append(prefix)
    
    # Sort prefixes for consistency
    for model_id in profile_map:
        profile_map[model_id] = sorted(profile_map[model_id])
    
    return profile_map


def get_inference_profile_arn(bedrock_client, model_id: str, profile_prefix: str) -> Optional[str]:
    """Get the ARN of a system-defined inference profile
    
    Args:
        bedrock_client: Boto3 Bedrock client
        model_id: Model ID
        profile_prefix: Profile prefix (us, eu, etc.)
        
    Returns:
        Profile ARN or None if not found
    """
    try:
        target_profile_id = f"{profile_prefix}.{model_id}"
        for profile in list_inference_profiles(bedrock_client, 'SYSTEM_DEFINED'):
            if profile.get('inferenceProfileId') == target_profile_id:
                return profile.get('inferenceProfileArn')
        return None
    except Exception as e:
        print(f"Error fetching inference profile: {e}", file=sys.stderr)
        return None


def create_application_inference_profile(bedrock_client, model_id: str, profile_prefix: Optional[str], region: str, profile_name: str) -> Optional[str]:
    """Create an application inference profile
    
    Args:
        bedrock_client: Boto3 Bedrock client
        model_id: Model ID
        profile_prefix: Profile prefix or None for base model
        region: AWS region
        profile_name: Name for the application profile
        
    Returns:
        Profile ARN or None if creation failed
    """
    try:
        # Determine source ARN
        if profile_prefix and profile_prefix != 'null':
            source_arn = get_inference_profile_arn(bedrock_client, model_id, profile_prefix)
            if not source_arn:
                print(f"Could not find system profile for {profile_prefix}.{model_id}", file=sys.stderr)
                return None
        else:
            # Base model ARN with correct partition
            source_arn = build_arn('bedrock', region, '', f"foundation-model/{model_id}")
        
        # Create application profile
        response = bedrock_client.create_inference_profile(
            inferenceProfileName=profile_name,
            modelSource={'copyFrom': source_arn}
        )
        
        return response.get('inferenceProfileArn')
        
    except Exception as e:
        print(f"Error creating application profile: {e}", file=sys.stderr)
        return None
