# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate quota index CSV for validation"""

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict
import sys

from bedrock_usage_analyzer.utils.yaml_handler import load_yaml, save_yaml
from bedrock_usage_analyzer.utils.csv_handler import write_csv
from bedrock_usage_analyzer.utils.paths import list_data_files, get_writable_path, get_bundle_path, get_data_path
from bedrock_usage_analyzer.aws.servicequotas import check_quota, QUOTA_OK, QUOTA_MISSING
from bedrock_usage_analyzer.aws.bedrock import get_regional_profile_prefixes
from bedrock_usage_analyzer.sync.quota_rules import mapping_conflict

logger = logging.getLogger(__name__)


class QuotaIndexGenerator:
    """Generates CSV index of all quota mappings for validation"""
    
    def __init__(self):
        self.models = {}
        self.entries = []
        self.error_entries = []
        self.mismatch_entries = []
        self.update_bundle = False
        # Filled by _cleanup_errors: per-(code, region) lookup results, regional prefixes,
        # and slots already known to contradict their model/endpoint
        self._region_checks = {}
        self._regional = set()
        self._mismatched = set()

    def run(self, update_bundle: bool = False):
        """Execute quota index generation
        
        Args:
            update_bundle: Also update bundled metadata (for maintainers)
        """
        self.update_bundle = update_bundle
        logger.info("Generating quota index for validation...\n")
        
        self._load_all_models()
        self._extract_quota_entries()
        self._fetch_quota_details()
        self._cleanup_errors()
        self._generate_csv()
        
        logger.info("\nQuota index generation complete!")
        logger.info("Review quota-index.csv to validate quota mappings")
    
    def _load_all_models(self):
        """Load all FM list files and merge endpoints from all regions"""
        fm_files = list_data_files('fm-list-*.yml')
        
        if not fm_files:
            logger.error("No fm-list files found")
            sys.exit(1)
        
        logger.info(f"Found {len(fm_files)} fm-list files")
        
        for fm_file in fm_files:
            # Extract region from filename
            filename = fm_file.name if hasattr(fm_file, 'name') else str(fm_file)
            region = filename.replace('fm-list-', '').replace('.yml', '')
            data = load_yaml(str(fm_file))
            
            for model in data.get('models', []):
                model_id = model['model_id']
                
                if model_id not in self.models:
                    # First time seeing this model - initialize
                    self.models[model_id] = {
                        'model_id': model_id,
                        'provider': model.get('provider'),
                        'inference_types': model.get('inference_types', []),
                        'inference_profiles': model.get('inference_profiles', []),
                        'endpoints': {}
                    }
                
                # Merge endpoints from this region, to the dictionary that aggregates all regions
                self._merge_endpoints(model_id, model, region)
        
        logger.info(f"Loaded {len(self.models)} unique models\n")
    
    def _merge_endpoints(self, model_id: str, model: Dict, region: str):
        """Merge endpoints from model into existing model entry"""
        new_endpoints = model.get('endpoints', {})
        
        for endpoint_type, endpoint_data in new_endpoints.items():
            existing_endpoints = self.models[model_id]['endpoints']
            
            if endpoint_type not in existing_endpoints:
                # New endpoint - add it
                existing_endpoints[endpoint_type] = {
                    **endpoint_data,
                    '_source_region': region
                }
            else:
                # Endpoint exists, potentially from other regions - check if new one has quotas
                existing_quotas = existing_endpoints[endpoint_type].get('quotas', {})
                new_quotas = endpoint_data.get('quotas', {})
                
                existing_has_quotas = any(v is not None for v in existing_quotas.values())
                new_has_quotas = any(v is not None for v in new_quotas.values())
                
                # Replace if new one has quotas and existing doesn't
                if new_has_quotas and not existing_has_quotas:
                    existing_endpoints[endpoint_type] = {
                        **endpoint_data,
                        '_source_region': region
                    }
    
    def _extract_quota_entries(self):
        """Extract all quota mappings from models"""
        # Avoid duplicate by listing only a unique combination of model ID, profile prefix, and metric/quota
        seen = set()
        
        for model_id, model in self.models.items():
            endpoints = model.get('endpoints', {})
            
            for endpoint_type, endpoint_data in endpoints.items():
                quotas = endpoint_data.get('quotas', {})
                source_region = endpoint_data.get('_source_region', 'unknown')
                
                for quota_type, quota_data in quotas.items():
                    # {code: L-xxx, name: "..."} or null
                    if quota_data and isinstance(quota_data, dict):
                        quota_code = quota_data.get('code')
                        quota_name = quota_data.get('name')
                        
                        if quota_code:
                            key = (model_id, endpoint_type, quota_type, quota_code)
                            if key not in seen:
                                seen.add(key)
                                self.entries.append({
                                    'model_id': model_id,
                                    'endpoint': endpoint_type,
                                    'quota_type': quota_type,
                                    'quota_code': quota_code,
                                    'quota_name': quota_name,
                                    'source_region': source_region
                                })
        
        logger.info(f"Found {len(self.entries)} unique quota mappings\n")
    
    def _fetch_quota_details(self):
        """Validate every mapped quota code against Service Quotas.

        Names are refreshed from the API. A code that Service Quotas reports as
        missing is marked ERROR and later removed from the fm-lists; other failures
        (throttling, network) leave the entry untouched.
        """
        if not self.entries:
            return

        logger.info(f"Validating {len(self.entries)} quota mappings against Service Quotas...\n")
        regional = set(get_regional_profile_prefixes())
        keys = sorted({(e['quota_code'], e['source_region']) for e in self.entries})
        # Independent lookups; a small pool keeps well under Service Quotas rate limits
        with ThreadPoolExecutor(max_workers=8) as pool:
            cache = dict(zip(keys, pool.map(lambda k: check_quota(*k), keys)))

        unverified = 0
        for entry in self.entries:
            status, quota = cache[(entry['quota_code'], entry['source_region'])]
            if status == QUOTA_OK:
                entry['quota_name'] = quota.get('QuotaName') or entry.get('quota_name') or 'N/A'
            # Checked for every entry, whatever the lookup outcome, with the best name known,
            # so the CSV and the fm-list cleanup always agree
            reason = mapping_conflict(entry['model_id'], entry['endpoint'], entry.get('quota_name'), regional)
            if reason:
                # The code belongs to another model or endpoint type
                logger.info(f"  Mismatch: {entry['model_id']} {entry['endpoint']} {entry['quota_type']} "
                            f"-> {entry['quota_code']} ({entry.get('quota_name')}): {reason}")
                entry['quota_name'] = 'MISMATCH'
                self.mismatch_entries.append(entry)
            elif status == QUOTA_MISSING:
                entry['previous_name'] = entry.get('quota_name')
                entry['quota_name'] = 'ERROR'
                self.error_entries.append(entry)
            elif status != QUOTA_OK:
                unverified += 1
                entry['quota_name'] = entry.get('quota_name') or 'N/A'
        if unverified:
            logger.info(f"  {unverified} mapping(s) could not be verified (API errors); kept as is")

    def _cleanup_errors(self):
        """Remove quota codes that do not exist, or belong to another model/endpoint, from every fm-list"""
        if not self.error_entries and not self.mismatch_entries:
            logger.info(f"\nThere is no erroneous entry.")
        else:
            logger.info(f"\nCleaning up {len(self.error_entries)} missing and "
                        f"{len(self.mismatch_entries)} mismatched entries...")
        stale = {(e['model_id'], e['endpoint'], e['quota_type'], e['quota_code']) for e in self.error_entries}
        # Quota availability differs by region: a code missing in the entry's source
        # region can exist elsewhere, so re-check it in every region before removing it
        self._region_checks = {
            (e['quota_code'], e['source_region']): QUOTA_MISSING for e in self.error_entries}
        self._regional = set(get_regional_profile_prefixes())
        self._mismatched = {(e['model_id'], e['endpoint'], e['quota_type'], e['quota_code'])
                            for e in self.mismatch_entries}
        regions = []
        for fm_file in list_data_files('fm-list-*.yml'):
            filename = fm_file.name if hasattr(fm_file, 'name') else str(fm_file)
            regions.append(filename.replace('fm-list-', '').replace('.yml', ''))
        # Look up, in parallel, every (code, region) pair the cleanup will ask about
        pending = sorted({(slot[3], region) for region in regions
                          for slot in self._stale_slots_in(region, stale)} - set(self._region_checks))
        with ThreadPoolExecutor(max_workers=8) as pool:
            for key, result in zip(pending, pool.map(lambda k: check_quota(*k), pending)):
                self._region_checks[key] = result[0]
        # Every region file is also checked for mismatches by the quota name stored with each
        # code, because the index itself samples only one source region per model endpoint
        for region in regions:
            self._cleanup_region_errors(region, stale)

        # A code confirmed in any region is still a valid mapping for the index
        confirmed = {code for (code, _), status in self._region_checks.items() if status == QUOTA_OK}
        for entry in list(self.error_entries):
            if entry['quota_code'] in confirmed:
                entry['quota_name'] = entry.get('previous_name') or 'N/A'
                self.error_entries.remove(entry)

    @staticmethod
    def _stale_slots_in(region: str, stale):
        """(model, endpoint, type, code) slots of one region file that are in ``stale``."""
        data = load_yaml(get_data_path(f'fm-list-{region}.yml')) or {}
        for model in data.get('models', []):
            for endpoint, endpoint_data in (model.get('endpoints') or {}).items():
                for quota_type, quota in ((endpoint_data or {}).get('quotas') or {}).items():
                    if isinstance(quota, dict):
                        slot = (model['model_id'], endpoint, quota_type, quota.get('code'))
                        if slot in stale:
                            yield slot

    def _missing_in(self, code: str, region: str) -> bool:
        key = (code, region)
        if key not in self._region_checks:
            self._region_checks[key] = check_quota(code, region)[0]
        return self._region_checks[key] == QUOTA_MISSING

    def _cleanup_region_errors(self, region: str, stale):
        """Null out codes missing in this region or contradicting their model/endpoint (user copy, else bundled)"""
        regional, mismatched = self._regional, self._mismatched
        yaml_file = get_writable_path(f'fm-list-{region}.yml')
        data = load_yaml(get_data_path(f'fm-list-{region}.yml')) or {}

        modified = False
        for model in data.get('models', []):
            for endpoint, endpoint_data in (model.get('endpoints') or {}).items():
                quotas = (endpoint_data or {}).get('quotas') or {}
                for quota_type, quota in quotas.items():
                    if not isinstance(quota, dict):
                        continue
                    code = quota.get('code')
                    slot = (model['model_id'], endpoint, quota_type, code)
                    reason = None
                    if slot in mismatched:
                        reason = 'mismatch'
                    elif mapping_conflict(model['model_id'], endpoint, quota.get('name'), regional):
                        reason = 'mismatch'
                    elif slot in stale and self._missing_in(code, region):
                        reason = 'missing'
                    if reason:
                        logger.info(f"  Removing {model['model_id']} -> {endpoint} -> {quota_type} ({code}) "
                                    f"in {region}: {reason}")
                        quotas[quota_type] = None
                        modified = True

        if modified:
            save_yaml(str(yaml_file), data)
            logger.info(f"  ✓ Updated {yaml_file}")

            if getattr(self, 'update_bundle', False):
                bundle_path = get_bundle_path()
                if bundle_path:
                    bundle_file = bundle_path / f'fm-list-{region}.yml'
                    save_yaml(str(bundle_file), data)
                    logger.info(f"  ✓ Updated {bundle_file} (bundled)")

    def _generate_csv(self):
        """Generate CSV file with valid entries"""
        valid_rows = [
            [e['model_id'], e['endpoint'], e['quota_type'], e['quota_code'], e['quota_name']]
            for e in self.entries if e.get('quota_name') not in ('ERROR', 'MISMATCH')
        ]
        
        output_file = get_writable_path('quota-index.csv')
        write_csv(
            str(output_file),
            ['model_id', 'endpoint', 'quota_type', 'quota_code', 'quota_name'],
            valid_rows
        )
        logger.info(f"\n✓ Generated {output_file} with {len(valid_rows)} valid entries")
        
        if getattr(self, 'update_bundle', False):
            bundle_path = get_bundle_path()
            if bundle_path:
                bundle_file = bundle_path / 'quota-index.csv'
                write_csv(
                    str(bundle_file),
                    ['model_id', 'endpoint', 'quota_type', 'quota_code', 'quota_name'],
                    valid_rows
                )
                logger.info(f"✓ Generated {bundle_file} (bundled)")
        
        if self.error_entries:
            logger.info(f"✓ Cleaned up {len(self.error_entries)} ERROR entries from YAML files")


def main():
    """Main entry point"""
    generator = QuotaIndexGenerator()
    generator.run()


if __name__ == "__main__":
    main()
