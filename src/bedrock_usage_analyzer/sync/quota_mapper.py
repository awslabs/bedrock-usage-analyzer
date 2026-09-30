# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Foundation model quota mapping using Bedrock LLM"""

import logging
import copy
import sys
from typing import Dict, List, Optional

from bedrock_usage_analyzer.utils.yaml_handler import load_yaml, save_yaml
from bedrock_usage_analyzer.utils.paths import get_data_path, get_writable_path, get_bundle_path
from bedrock_usage_analyzer.aws.servicequotas import fetch_service_quotas
from bedrock_usage_analyzer.aws.bedrock_llm import extract_common_name, extract_quota_codes
from bedrock_usage_analyzer.aws.bedrock import get_endpoint_quota_keywords, get_regional_profile_prefixes
from bedrock_usage_analyzer.sync.quota_rules import mapping_conflict

logger = logging.getLogger(__name__)

def _normalize(text: str) -> str:
    """Lower-case and collapse '-', '_' and spaces so model IDs match quota names."""
    return ' '.join(text.lower().replace('-', ' ').replace('_', ' ').split())


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
        
    def run(self, update_bundle: bool = False):
        """Execute quota mapping for all regions
        
        Args:
            update_bundle: Also update bundled metadata (for maintainers)
        """
        self.update_bundle = update_bundle
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
        from bedrock_usage_analyzer.sync.regions import load_region_names, regions_for_credentials
        # Quota codes are cached across regions, so stay inside the credentials' partition
        all_regions, _ = regions_for_credentials(load_region_names(), self.bedrock_region)

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
        if quotas is None:
            # Listing failed (throttling, pagination error, service not in region):
            # leave this region's saved mappings untouched
            logger.info("  ⊘ Could not list service quotas, skipping region\n")
            return
        logger.info(f"  Found {len(quotas)} quotas")
        
        fm_list = self._load_fm_list(region)
        if not fm_list:
            logger.info(f"  ⊘ No FM list found, skipping\n")
            return
        
        logger.info(f"  Mapping quotas for {len(fm_list)} models...")
        
        updated_count = 0
        regional = set(get_regional_profile_prefixes())
        for i, fm in enumerate(fm_list, 1):
            model_id = fm['model_id']
            logger.info(f"    [{i}/{len(fm_list)}] {model_id}... ", extra={'end': ''})
            
            endpoints_to_process = self._get_endpoints_to_process(fm)
            # First, before any 'continue': saved codes that contradict their model/endpoint
            # (written by older versions) go even when nothing new replaces them
            self._drop_conflicting_saved_codes(fm, regional)

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
            
            # Merge per metric. A metric without a new match keeps its saved code: "no match"
            # cannot be told apart from a failed LLM call or a keyword miss, and wiping a
            # correct code is worse than keeping a stale one. `bua refresh quota-index` removes
            # codes that Service Quotas rejects or that contradict their model/endpoint.
            endpoints = fm.setdefault('endpoints', {})
            for endpoint_type, new in endpoints_data.items():
                saved = (endpoints.get(endpoint_type) or {}).get('quotas') or {}
                merged = dict(saved)
                merged.update({metric: value for metric, value in new['quotas'].items() if value})
                for metric in new['quotas']:
                    merged.setdefault(metric, None)
                endpoints[endpoint_type] = {**(endpoints.get(endpoint_type) or {}), 'quotas': merged}

            if endpoints_data:
                updated_count += 1
                endpoint_summary = ', '.join(endpoints_data.keys())
                logger.info(f"✓ ({endpoint_summary})")
            else:
                logger.info("✗ (no mappings)")
        
        self._save_fm_list(region, fm_list)
        logger.info(f"  ✓ Updated {updated_count} models\n")
    
    @staticmethod
    def _drop_conflicting_saved_codes(fm: Dict, regional) -> None:
        for endpoint_type, endpoint_value in (fm.get('endpoints') or {}).items():
            saved = (endpoint_value or {}).get('quotas') or {}
            for metric, value in list(saved.items()):
                if isinstance(value, dict) and mapping_conflict(fm['model_id'], endpoint_type, value.get('name'), regional):
                    saved[metric] = None

    def _get_endpoints_to_process(self, fm: Dict) -> List[str]:
        """Determine which endpoints to process for a model"""
        # Simply return the keys from the endpoints dict
        return list(fm.get('endpoints', {}).keys())
    
    def _get_quota_mapping(self, region: str, model_id: str, common_name: str, 
                          endpoint_type: str, quotas: List[Dict]) -> Optional[Dict]:
        """Get quota mapping for a specific endpoint"""
        cache_key = (model_id, endpoint_type if endpoint_type in ['base', 'cross-region', 'global'] else 'cross-region')
        region_codes = {q.get('QuotaCode') for q in quotas}
        cached = self.lcode_cache.get(cache_key)
        # Codes are shared across regions, but a region may lack a quota: reuse the cached
        # mapping only when every code in it exists in this region
        if cached and all(v['code'] in region_codes for v in cached.values() if v):
            return copy.deepcopy(cached)

        # Get the candidates (list) of possible quota names for a given FM, based on the keyword search on the FM's common or base name
        matching_quotas = self._find_matching_quotas(quotas, common_name, endpoint_type, model_id)
        if not matching_quotas:
            return None

        # Call LLM
        # Inputs are the possible matching quota names for the given FM
        # Outputs are the mapped quotas for each metrics (e.g. TPM, TPD, RPM, concurrent)
        quota_mapping = extract_quota_codes(
            self.bedrock_region, self.model_id, model_id,
            endpoint_type, matching_quotas
        )
        quota_mapping = self._drop_invalid_choices(quota_mapping, matching_quotas, model_id, endpoint_type)

        if quota_mapping and cache_key not in self.lcode_cache:
            self.lcode_cache[cache_key] = quota_mapping

        return quota_mapping

    @staticmethod
    def _drop_invalid_choices(quota_mapping, candidates, model_id, endpoint_type):
        """Reject LLM picks outside the candidate list or contradicting the model/endpoint."""
        if not quota_mapping:
            return quota_mapping
        by_code = {c['code']: c['name'] for c in candidates}
        regional = set(get_regional_profile_prefixes())
        cleaned = {}
        for metric, choice in quota_mapping.items():
            if choice and choice.get('code') in by_code and not mapping_conflict(
                    model_id, endpoint_type, by_code[choice['code']], regional):
                cleaned[metric] = {'code': choice['code'], 'name': by_code[choice['code']]}
            else:
                if choice:
                    logger.debug(f"Rejected {choice.get('code')} for {model_id} ({endpoint_type})")
                cleaned[metric] = None
        return cleaned if any(cleaned.values()) else None

    def _find_matching_quotas(self, quotas: List[Dict], common_name: str, endpoint_type: str,
                              model_id: Optional[str] = None) -> List[Dict]:
        """Find quotas matching the common name and endpoint type"""
        matching = []

        endpoint_quota_keywords = get_endpoint_quota_keywords()
        required_keyword = endpoint_quota_keywords.get(endpoint_type)
        if not required_keyword:
            return matching
        regional = set(get_regional_profile_prefixes())

        # Perform keyword search to find the potential quotas for a given base/common name of an FM.
        # Hyphens and spaces are treated alike: 'gpt-oss' must match "GPT OSS Safeguard 20B".
        name_key = _normalize(common_name)
        for quota in quotas:
            quota_name = quota.get('QuotaName', '').lower()

            # The first term below performs keyword matching "Does the quota name contain this FM common/base name?" operation
            if not (name_key in _normalize(quota_name) and required_keyword in quota_name):
                continue
            # 'cross-region' also matches "Global cross-region ..." and a family name matches
            # every version: drop candidates that contradict this endpoint or model version
            if model_id and mapping_conflict(model_id, endpoint_type, quota['QuotaName'], regional):
                continue
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
            fm_file = get_data_path(f'fm-list-{region}.yml')
            data = load_yaml(fm_file)
            return data.get('models', [])
        except Exception:
            return None
    
    def _save_fm_list(self, region: str, fm_list: List[Dict]):
        """Save FM list for region"""
        output_file = get_writable_path(f'fm-list-{region}.yml')
        data = {'models': fm_list}
        save_yaml(str(output_file), data)
        logger.info(f"  ✓ Saved: {output_file}")
        
        if getattr(self, 'update_bundle', False):
            bundle_path = get_bundle_path()
            if bundle_path:
                bundle_file = bundle_path / f'fm-list-{region}.yml'
                save_yaml(str(bundle_file), data)
                logger.info(f"  ✓ Saved: {bundle_file} (bundled)")
