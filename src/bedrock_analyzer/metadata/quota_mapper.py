# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Foundation model quota mapping using Bedrock LLM"""

import logging
import copy
import sys
from typing import Dict, List, Optional

from bedrock_analyzer.utils.yaml_handler import load_yaml, save_yaml
from bedrock_analyzer.aws.servicequotas import fetch_service_quotas
from bedrock_analyzer.aws.bedrock_llm import extract_common_name, extract_quota_codes
from bedrock_analyzer.aws.bedrock import get_endpoint_quota_keywords

logger = logging.getLogger(__name__)


class QuotaMapper:
    """Maps foundation models to their service quotas using Bedrock LLM"""
    
    def __init__(self, bedrock_region: str, model_id: str, target_region: Optional[str] = None):
        """Initialize quota mapper
        
        Args:
            bedrock_region: AWS region for Bedrock API calls
            model_id: Model ID to use for intelligent mapping
            target_region: Optional specific region to process
        """
        self.bedrock_region = bedrock_region
        self.model_id = model_id
        self.target_region = target_region
        self.common_name_cache = {}
        self.lcode_cache = {}
        
    def run(self):
        """Execute quota mapping for all regions"""
        logger.info(f"Using model: {self.model_id}")
        logger.info(f"Bedrock region: {self.bedrock_region}")
        if self.target_region:
            logger.info(f"Target region: {self.target_region}\n")
        
        regions = self._get_regions_to_process()
        logger.info(f"Processing {len(regions)} region(s)...\n")
        
        for region in regions:
            self._process_region(region)
    
    def _get_regions_to_process(self) -> List[str]:
        """Get list of regions to process"""
        regions_data = load_yaml('metadata/regions.yml')
        all_regions = regions_data.get('regions', [])
        
        if self.target_region:
            if self.target_region not in all_regions:
                logger.error(f"Region '{self.target_region}' not found")
                sys.exit(1)
            # If region argument is passed, then process only that particular region
            return [self.target_region]
        
        return all_regions
    
    def _process_region(self, region: str):
        """Process quota mapping for a single region"""
        logger.info(f"Region: {region}")
        
        quotas = fetch_service_quotas(region)
        logger.info(f"  Found {len(quotas)} quotas")
        
        fm_list = self._load_fm_list(region)
        if not fm_list:
            logger.info(f"  ⊘ No FM list found, skipping\n")
            return
        
        logger.info(f"  Mapping quotas for {len(fm_list)} models...")
        
        updated_count = 0
        for i, fm in enumerate(fm_list, 1):
            model_id = fm['model_id']
            logger.info(f"    [{i}/{len(fm_list)}] {model_id}... ", extra={'end': ''})
            
            endpoints_to_process = self._get_endpoints_to_process(fm)
            
            if not endpoints_to_process:
                logger.info("⊘ (no endpoints)")
                continue
            
            # Call LLM
            # For a given model get the common/base name, so that the keyword search later is not too specific to cause false negative, and not too broad to cost much tokens
            common_name = self._get_common_name(model_id)
            if not common_name:
                logger.info("✗ (no common name)")
                continue
            
            endpoints_data = {}
            for endpoint_type in endpoints_to_process:
                # Get the mapping between the current FM with the matching quotas for its RPM, TPM, TPD, concurrent invocations (if available)
                quota_mapping = self._get_quota_mapping(
                    region, model_id, common_name, endpoint_type, quotas
                )
                if quota_mapping:
                    endpoints_data[endpoint_type] = {'quotas': quota_mapping}
            
            if endpoints_data:
                fm['endpoints'] = endpoints_data
                updated_count += 1
                endpoint_summary = ', '.join(endpoints_data.keys())
                logger.info(f"✓ ({endpoint_summary})")
            else:
                logger.info("✗ (no mappings)")
        
        self._save_fm_list(region, fm_list)
        logger.info(f"  ✓ Updated {updated_count} models\n")
    
    def _get_endpoints_to_process(self, fm: Dict) -> List[str]:
        """Determine which endpoints to process for a model"""
        # Simply return the keys from the endpoints dict
        return list(fm.get('endpoints', {}).keys())
    
    def _get_quota_mapping(self, region: str, model_id: str, common_name: str, 
                          endpoint_type: str, quotas: List[Dict]) -> Optional[Dict]:
        """Get quota mapping for a specific endpoint"""
        cache_key = (model_id, endpoint_type if endpoint_type in ['base', 'cross-region', 'global'] else 'cross-region')
        if cache_key in self.lcode_cache:
            return copy.deepcopy(self.lcode_cache[cache_key])
        
        # Get the candidates (list) of possible quota names for a given FM, based on the keyword search on the FM's common or base name
        matching_quotas = self._find_matching_quotas(quotas, common_name, endpoint_type)
        if not matching_quotas:
            return None
        
        # Call LLM
        # Inputs are the possible matching quota names for the given FM
        # Outputs are the mapped quotas for each metrics (e.g. TPM, TPD, RPM, concurrent)
        quota_mapping = extract_quota_codes(
            self.bedrock_region, self.model_id, model_id,
            endpoint_type, matching_quotas
        )
        
        if quota_mapping:
            self.lcode_cache[cache_key] = quota_mapping
        
        return quota_mapping
    
    def _find_matching_quotas(self, quotas: List[Dict], common_name: str, endpoint_type: str) -> List[Dict]:
        """Find quotas matching the common name and endpoint type"""
        matching = []
        
        endpoint_quota_keywords = get_endpoint_quota_keywords()
        required_keyword = endpoint_quota_keywords.get(endpoint_type)
        if not required_keyword:
            return matching
        
        # Perform keyword search to find the potential quotas for a given base/common name of an FM
        for quota in quotas:
            quota_name = quota.get('QuotaName', '').lower()
            
            # The first term below performs keyword matching "Does the quota name contain this FM common/base name?" operation
            if common_name in quota_name and required_keyword in quota_name:
                matching.append({
                    'name': quota['QuotaName'],
                    'code': quota['QuotaCode'],
                    'value': quota.get('Value', 0)
                })
        
        return matching
    
    def _get_common_name(self, model_id: str) -> Optional[str]:
        """Get common name for model (with caching)"""
        if model_id in self.common_name_cache:
            return self.common_name_cache[model_id]
        
        common_name = extract_common_name(self.bedrock_region, self.model_id, model_id)
        if common_name:
            self.common_name_cache[model_id] = common_name
        
        return common_name
    
    def _load_fm_list(self, region: str) -> Optional[List[Dict]]:
        """Load FM list for region"""
        try:
            data = load_yaml(f'metadata/fm-list-{region}.yml')
            return data.get('models', [])
        except Exception:
            return None
    
    def _save_fm_list(self, region: str, fm_list: List[Dict]):
        """Save FM list for region"""
        save_yaml(f'metadata/fm-list-{region}.yml', {'models': fm_list})
