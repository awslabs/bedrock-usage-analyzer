# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inference profile discovery for Bedrock models"""

import logging

from bedrock_analyzer.aws.bedrock import get_default_region_prefix_map

logger = logging.getLogger(__name__)


class InferenceProfileFetcher:
    """Handles inference profile discovery"""
    
    def __init__(self, bedrock_client):
        self.bedrock_client = bedrock_client
        self.prefix_map = get_default_region_prefix_map()
        self._all_profiles_cache = None  # Cache for all profiles
    
    def find_profiles(self, model_id, profile_prefix):
        """Find system-defined profile and all application profiles based on it
        
        Returns:
            tuple: (profiles list, profile_names dict, profile_metadata dict)
                   profile_metadata contains 'id' and 'tags' for each profile
        """
        logger.info(f"  Discovering inference profiles...")
        
        if profile_prefix is None:
            target_endpoint = model_id
        else:
            target_endpoint = f"{profile_prefix}.{model_id}"
        
        profiles = [target_endpoint]
        profile_names = {target_endpoint: target_endpoint}  # System profile uses ID as name
        profile_metadata = {target_endpoint: {'id': 'N/A', 'tags': {}}}  # System profiles have no ID/tags
        
        # Fetch APPLICATION profiles explicitly (API returns only SYSTEM by default)
        # Use cache if available
        if self._all_profiles_cache is None:
            logger.info(f"  Calling list_inference_profiles API (fetching all profiles)...")
            self._all_profiles_cache = []
            response = self.bedrock_client.list_inference_profiles(
                maxResults=1000,
                typeEquals='APPLICATION'
            )
            
            pagination_count = 1
            
            # Handle pagination and cache all profiles
            while True:
                self._all_profiles_cache.extend(response['inferenceProfileSummaries'])
                
                if 'nextToken' in response:
                    pagination_count += 1
                    logger.info(f"    Fetching page {pagination_count}...")
                    response = self.bedrock_client.list_inference_profiles(
                        maxResults=1000,
                        typeEquals='APPLICATION',
                        nextToken=response['nextToken']
                    )
                else:
                    break
            
            logger.info(f"  Cached {len(self._all_profiles_cache)} application profiles")
        else:
            logger.info(f"  Using cached profiles ({len(self._all_profiles_cache)} profiles)")
        
        matched_profiles = 0
        
        # Search cached profiles
        for profile in self._all_profiles_cache:
            model_arns = [m['modelArn'] for m in profile['models']]
            source = self._infer_source_profile(model_arns)
            
            if source == target_endpoint:
                matched_profiles += 1
                profile_id = profile['inferenceProfileId']
                profiles.append(profile_id)
                profile_names[profile_id] = profile.get('inferenceProfileName', profile_id)
                
                # Get inferenceProfileId directly from response
                inference_profile_id = profile.get('inferenceProfileId', 'N/A')
                
                # Fetch tags via list_tags_for_resource API
                tags = {}
                profile_arn = profile.get('inferenceProfileArn')
                if profile_arn:
                    try:
                        tags_response = self.bedrock_client.list_tags_for_resource(resourceARN=profile_arn)
                        # Convert list of {key, value} dicts to simple dict
                        tags = {tag['key']: tag['value'] for tag in tags_response.get('tags', [])}
                    except Exception as e:
                        logger.info(f"  Warning: Could not fetch tags for {profile_id}: {e}")
                    profile_metadata[profile_id] = {
                        'id': inference_profile_id,
                        'tags': tags
                    }
        
        logger.info(f"  Profile discovery: {matched_profiles} application profiles matched")
        return profiles, profile_names, profile_metadata
    
    def _infer_source_profile(self, model_arns):
        """Infer which endpoint an application profile is based on"""
        # Extract model ID from first ARN (all ARNs have same model)
        model_id = model_arns[0].split('/')[-1]
        
        if len(model_arns) == 1:
            # On-demand: single model ARN
            return model_id
        
        # System profile: multiple model ARNs across regions
        regions = [arn.split(':')[3] for arn in model_arns]
        region_prefixes = set(r.split('-')[0] for r in regions)
        
        if len(region_prefixes) == 1:
            # Single region prefix (us, eu, ap, ca, jp, au)
            region_prefix = list(region_prefixes)[0]
            system_prefix = self.prefix_map.get(region_prefix, region_prefix)
            return f"{system_prefix}.{model_id}"
        else:
            # Mixed regions = global
            return f"global.{model_id}"
