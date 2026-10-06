# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""AWS Bedrock service operations"""

import sys
import logging
from typing import List, Dict, Optional, Tuple

import yaml

from bedrock_usage_analyzer.aws.client_factory import create_client
from bedrock_usage_analyzer.core.errors import is_access_denied
from bedrock_usage_analyzer.utils.partition import build_arn, partition_region_prefix

logger = logging.getLogger(__name__)


# Quota keyword constants
QUOTA_KEYWORD_ON_DEMAND = 'on-demand'
QUOTA_KEYWORD_CROSS_REGION = 'cross-region'
QUOTA_KEYWORD_GLOBAL = 'global'
QUOTA_KEYWORD_CUSTOM = 'custom model deployment'

# The fm-list endpoint holding a base model's on-demand custom model deployment quotas
# ("(Model customization) Sum of on demand custom model deployment tokens per minute for
# Amazon Nova Lite"). Not an inference profile prefix: 'custom.<model>' is never invoked.
CUSTOM_ENDPOINT = 'custom'

# Used only when no prefix-mapping.yml can be read at all (the bundled file is the source
# of truth and normally always present)
FALLBACK_PROFILE_PREFIXES = frozenset({'us', 'eu', 'apac', 'jp', 'au', 'ca', 'in', 'us-gov', 'global'})

# Cache for prefix mapping to avoid repeated file reads
_prefix_mapping_cache = None
_profile_prefixes_cache = None  # (mapping it was built from, frozenset)


def _valid_prefix_entries(data, source) -> List[Dict]:
    """The usable entries of parsed prefix-mapping.yml ``data``: mappings with a string
    'prefix' and 'quota_keyword'. Anything else in a hand-edited file is skipped with a
    warning, so the other layer (or the fallback set) still applies."""
    if not data:
        return []  # empty file
    entries = data.get('prefixes') if isinstance(data, dict) else None
    if not isinstance(entries, list):
        if entries is not None or not isinstance(data, dict):
            logger.warning(f"Ignoring {source}: expected 'prefixes:' with a list of entries")
        return []
    valid = [e for e in entries if isinstance(e, dict) and isinstance(e.get('prefix'), str)
             and isinstance(e.get('quota_keyword'), str)]
    if len(valid) != len(entries):
        logger.warning(f"Ignoring {len(entries) - len(valid)} malformed entr(ies) in {source} "
                       f"(each needs 'prefix' and 'quota_keyword')")
    return valid


def read_prefix_file(path) -> List[Dict]:
    from bedrock_usage_analyzer.utils.yaml_handler import load_yaml
    try:
        data = load_yaml(str(path))
    except (FileNotFoundError, OSError):
        return []
    except yaml.YAMLError as e:
        # A hand-edited file with a syntax error: use the other layer (or the fallback set)
        logger.warning(f"Ignoring unreadable {path}: {e}")
        return []
    return _valid_prefix_entries(data, path)


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
        if not _prefix_mapping_cache:
            raise FileNotFoundError("prefix-mapping.yml not found")  # known missing: no re-read
        return _prefix_mapping_cache

    bundled, user = prefix_mapping_layers()
    merged: Dict[str, Dict] = {}
    for entry in bundled + user:
        merged[entry['prefix']] = entry

    if not merged:
        # Remembered (an empty list) until load_prefix_mapping(refresh=True), so the fallback
        # prefixes are used without reading both files on every call
        _prefix_mapping_cache = []
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
    bundled = _valid_prefix_entries(load_bundled_yaml('prefix-mapping.yml'), 'bundled prefix-mapping.yml')
    return bundled, list(read_prefix_file(get_user_data_dir() / 'prefix-mapping.yml'))


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
    try:
        mapping = _load_prefix_mapping()
    except FileNotFoundError:
        mapping = []
    if not mapping:
        # No prefix-mapping.yml at all: the keywords of the prefixes this release knows
        keywords = {'base': QUOTA_KEYWORD_ON_DEMAND, 'global': QUOTA_KEYWORD_GLOBAL,
                    **{p: QUOTA_KEYWORD_CROSS_REGION for p in FALLBACK_PROFILE_PREFIXES - {'global'}}}
    else:
        keywords = {m['prefix']: m['quota_keyword'] for m in mapping}
    keywords.setdefault(CUSTOM_ENDPOINT, QUOTA_KEYWORD_CUSTOM)  # built in, not in prefix-mapping.yml
    return keywords


def get_endpoint_descriptions() -> Dict[str, str]:
    """Get mapping of endpoint prefix to description
    
    Returns:
        Dict mapping prefix to description (e.g., {'base': 'on-demand', 'us': 'cross-region inference profile'})
    """
    try:
        mapping = _load_prefix_mapping()
    except FileNotFoundError:
        mapping = []
    if not mapping:
        # No prefix-mapping.yml at all: describe the prefixes this release knows
        descriptions = {'base': 'on-demand', 'global': 'global inference profile',
                        **{p: 'cross-region inference profile' for p in FALLBACK_PROFILE_PREFIXES - {'global'}}}
    else:
        descriptions = {m['prefix']: m.get('description') or m['prefix'] for m in mapping}
    descriptions.setdefault(CUSTOM_ENDPOINT, 'on-demand custom model deployment')
    return descriptions


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
    # Same source and fallback as get_regional_profile_prefixes
    result = {prefix: prefix for prefix in get_regional_profile_prefixes()}
    result['ap'] = 'apac'  # Special case: 'ap' region prefix maps to 'apac' system profile
    return result


def get_profile_prefixes() -> frozenset:
    """All system inference profile prefixes, from prefix-mapping.yml (fallback set without it)."""
    global _profile_prefixes_cache
    try:
        mapping = _load_prefix_mapping()
    except FileNotFoundError:
        return FALLBACK_PROFILE_PREFIXES
    # Keyed by the mapping list itself: the loader returns the same cached list until it is
    # reloaded, so a reload (or a stubbed loader) always rebuilds the set
    if _profile_prefixes_cache is None or _profile_prefixes_cache[0] is not mapping:
        mapped = frozenset(m['prefix'] for m in mapping if m.get('prefix') != 'base')
        _profile_prefixes_cache = (mapping, mapped or FALLBACK_PROFILE_PREFIXES)
    return _profile_prefixes_cache[1]


def endpoint_id(model_id: str, prefix: Optional[str] = None) -> str:
    """Endpoint ID of a model: '<prefix>.<model>' for a profile, the model ID for its base
    (on-demand) endpoint, written as prefix None or 'base'. The inverse of split_profile_id."""
    return model_id if prefix in (None, 'base') else f"{prefix}.{model_id}"


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


def region_group(region: str) -> str:
    """Region family used to guess a profile prefix.

    'eu-west-1' -> 'eu', 'us-gov-west-1' -> 'us-gov', 'us-iso-east-1' -> 'us-iso',
    so regions of other partitions never fall into a commercial family.
    """
    # The partition table decides (one place for new partitions); commercial: first segment
    return partition_region_prefix(region) or region.split('-')[0]


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
            profile_id = profile.get('inferenceProfileId') or ''
            if profile.get('type') == 'SYSTEM_DEFINED' and '.' in profile_id:
                system_prefix = profile_id.split('.')[0]
                
                # Skip if already processed or if it's 'global'
                if system_prefix in seen_prefixes or system_prefix == 'global':
                    continue
                
                model_arns = [m.get('modelArn', '') for m in profile.get('models') or []]
                
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
            entry = {
                'model_id': model['modelId'],
                'provider': model['providerName'],
                'inference_types': model.get('inferenceTypesSupported', [])
            }
            if model.get('customizationsSupported'):
                # Custom models of it can be deployed on demand: their quotas are its 'custom' endpoint
                entry['customizations'] = list(model['customizationsSupported'])
            models.append(entry)
        
        return models
    
    except Exception as e:
        error_msg = str(e)
        # The shared permission rule, plus the opt-in messages of a region not enabled
        if is_access_denied(e) or any(x in error_msg for x in ['not enabled', 'not subscribed']):
            print(f"  ⊘ Skipping {region} (access denied or not enabled)", file=sys.stderr)
        else:
            print(f"  ✗ Failed to fetch models for {region}: {e}", file=sys.stderr)
        return None


def list_system_profiles(region: str) -> Optional[List[Dict]]:
    """The region's system inference profiles, or None when the listing failed (not "none listed")."""
    try:
        return list_inference_profiles(create_client('bedrock', region), 'SYSTEM_DEFINED')
    except Exception as e:
        # Inference profiles might not be available in all regions
        logger.warning(f"  Could not list inference profiles in {region}: {e}")
        return None


def fetch_all_inference_profiles(region: str) -> List[Dict]:
    """Fetch ALL inference profiles in region
    This fetches only system inference profile, not application inference profile
    The purpose is to list down the available system inference profiles for a given FM.
    
    Args:
        region: AWS region name
        
    Returns:
        List of inference profile dictionaries; [] on failure (the contract from earlier
        releases; list_system_profiles tells a failure apart)
    """
    return list_system_profiles(region) or []


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
        for model in profile.get('models') or []:
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
