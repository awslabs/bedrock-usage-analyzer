# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""AWS Bedrock service operations"""

import boto3
import sys
import os
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


# Quota keyword constants
QUOTA_KEYWORD_ON_DEMAND = 'on-demand'
QUOTA_KEYWORD_CROSS_REGION = 'cross-region'
QUOTA_KEYWORD_GLOBAL = 'global'

# Cache for prefix mapping to avoid repeated file reads
_prefix_mapping_cache = None


def _load_prefix_mapping() -> List[Dict]:
    """Load prefix mapping from metadata file or discover if missing
    
    Returns:
        List of prefix mapping dictionaries
        
    Raises:
        FileNotFoundError: If metadata/prefix-mapping.yml doesn't exist
    """
    global _prefix_mapping_cache
    
    if _prefix_mapping_cache is not None:
        return _prefix_mapping_cache
    
    metadata_file = 'metadata/prefix-mapping.yml'
    
    if os.path.exists(metadata_file):
        from bedrock_analyzer.utils.yaml_handler import load_yaml
        data = load_yaml(metadata_file)
        _prefix_mapping_cache = data.get('prefixes', [])
        return _prefix_mapping_cache
    else:
        # File missing - error out and ask user to refresh
        raise FileNotFoundError(
            f"\n{metadata_file} not found!\n"
            f"Please run: ./bin/refresh-fm-list\n"
            f"This will refresh both foundation model lists and prefix mapping."
        )


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
    mapping = _load_prefix_mapping()
    return [m['prefix'] for m in mapping if m['is_regional']]


def get_default_region_prefix_map() -> Dict[str, str]:
    """Get mapping of region prefix to system profile prefix
    
    Returns:
        Dict mapping region prefix to system profile prefix (e.g., {'us': 'us', 'ap': 'apac'})
    """
    mapping = _load_prefix_mapping()
    result = {m['prefix']: m['prefix'] for m in mapping if m['is_regional']}
    result['ap'] = 'apac'  # Special case: 'ap' region prefix maps to 'apac' system profile
    return result


def discover_prefix_mapping(region: str) -> List[Dict]:
    """Discover system profile prefixes from Bedrock API
    
    Discovers regional inference profile prefixes (us, eu, jp, au, apac, ca, etc.)
    by analyzing SYSTEM_DEFINED profiles. Automatically classifies as regional
    if model ARNs span multiple regions with same prefix.
    
    Args:
        region: AWS region to use for API calls
        
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
        bedrock = boto3.client('bedrock', region_name=region)
        response = bedrock.list_inference_profiles(maxResults=1000)
        
        # Collect all profiles with pagination
        all_profiles = []
        while True:
            all_profiles.extend(response['inferenceProfileSummaries'])
            if 'nextToken' in response:
                response = bedrock.list_inference_profiles(
                    maxResults=1000,
                    nextToken=response['nextToken']
                )
            else:
                break
        
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
                    regions = [arn.split(':')[3] for arn in model_arns]
                    region_prefixes = set(r.split('-')[0] for r in regions)
                    
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
        bedrock = boto3.client('bedrock', region_name=region)
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
        bedrock = boto3.client('bedrock', region_name=region)
        
        # Use paginator to handle large result sets
        paginator = bedrock.get_paginator('list_inference_profiles')
        all_profiles = []
        
        for page in paginator.paginate():
            all_profiles.extend(page.get('inferenceProfileSummaries', []))
        
        return all_profiles
    
    except Exception as e:
        # Inference profiles might not be available in all regions
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
        
        # Extract prefix (us, eu, jp, au, apac, global)
        if '.' not in profile_id:
            continue
        prefix = profile_id.split('.')[0]
        
        # Add this prefix to all models in this profile
        for model in profile.get('models', []):
            model_arn = model.get('modelArn', '')
            
            # Extract model_id from ARN (format: arn:aws:bedrock:region::foundation-model/model-id)
            if ':foundation-model/' in model_arn:
                model_id = model_arn.split(':foundation-model/')[-1]
                
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
        next_token = None
        
        while True:
            params = {'maxResults': 1000}
            if next_token:
                params['nextToken'] = next_token
            
            response = bedrock_client.list_inference_profiles(**params)
            
            for profile in response.get('inferenceProfileSummaries', []):
                if profile.get('inferenceProfileId') == target_profile_id:
                    return profile.get('inferenceProfileArn')
            
            next_token = response.get('nextToken')
            if not next_token:
                break
        
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
            # Base model ARN
            source_arn = f"arn:aws:bedrock:{region}::foundation-model/{model_id}"
        
        # Create application profile
        response = bedrock_client.create_inference_profile(
            inferenceProfileName=profile_name,
            modelSource={'copyFrom': source_arn}
        )
        
        return response.get('inferenceProfileArn')
        
    except Exception as e:
        print(f"Error creating application profile: {e}", file=sys.stderr)
        return None
