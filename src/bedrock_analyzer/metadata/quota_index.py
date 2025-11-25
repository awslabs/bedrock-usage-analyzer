# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate quota index CSV for validation"""

import glob
import logging
from typing import Dict, List, Set
import sys

from bedrock_analyzer.utils.yaml_handler import load_yaml, save_yaml
from bedrock_analyzer.utils.csv_handler import write_csv
from bedrock_analyzer.aws.servicequotas import get_quota_details

logger = logging.getLogger(__name__)


class QuotaIndexGenerator:
    """Generates CSV index of all quota mappings for validation"""
    
    def __init__(self):
        self.models = {}
        self.entries = []
        self.error_entries = []
    
    def run(self):
        """Execute quota index generation"""
        logger.info("Generating quota index for validation...\n")
        
        self._load_all_models()
        self._extract_quota_entries()
        self._fetch_quota_details()
        self._cleanup_errors()
        self._generate_csv()
        
        logger.info("\nQuota index generation complete!")
        logger.info("Review metadata/quota-index.csv to validate quota mappings")
    
    def _load_all_models(self):
        """Load all FM list files and merge endpoints from all regions"""
        fm_files = glob.glob('metadata/fm-list-*.yml')
        
        if not fm_files:
            logger.error("No fm-list files found in metadata/")
            sys.exit(1)
        
        logger.info(f"Found {len(fm_files)} fm-list files")
        
        for fm_file in fm_files:
            region = fm_file.replace('metadata/fm-list-', '').replace('.yml', '')
            data = load_yaml(fm_file)
            
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
        """Fetch quota details from AWS (skipped if names already present)"""
        if not self.entries:
            return
        
        # Check if we already have quota names (new format)
        entries_without_names = [e for e in self.entries if not e.get('quota_name')]
        
        if not entries_without_names:
            logger.info(f"All {len(self.entries)} entries already have quota names (new format)\n")
            return
        
        logger.info(f"Fetching quota details for {len(entries_without_names)} entries without names...\n")
        
        for entry in entries_without_names:
            quota_code = entry['quota_code']
            region = entry['source_region']
            
            quota = get_quota_details(quota_code, region)
            
            if quota:
                entry['quota_name'] = quota.get('QuotaName', 'N/A')
            else:
                entry['quota_name'] = 'ERROR'
                self.error_entries.append(entry)
    
    def _cleanup_errors(self):
        """Remove ERROR entries from YAML files"""
        if not self.error_entries:
            logger.info(f"\nThere is no erroneous entry.")
            return
        
        logger.info(f"\nCleaning up {len(self.error_entries)} ERROR entries...")
        
        # Group by region
        by_region = {}
        for entry in self.error_entries:
            region = entry['source_region']
            if region not in by_region:
                by_region[region] = []
            by_region[region].append(entry)
        
        # Update each region's YAML
        for region, entries in by_region.items():
            self._cleanup_region_errors(region, entries)
    
    def _cleanup_region_errors(self, region: str, entries: List[Dict]):
        """Clean up errors for a specific region"""
        yaml_file = f'metadata/fm-list-{region}.yml'
        data = load_yaml(yaml_file)
        
        modified = False
        for entry in entries:
            model_id = entry['model_id']
            endpoint = entry['endpoint']
            quota_type = entry['quota_type']
            
            for model in data.get('models', []):
                if model['model_id'] == model_id:
                    if 'endpoints' in model and endpoint in model['endpoints']:
                        if 'quotas' in model['endpoints'][endpoint]:
                            if quota_type in model['endpoints'][endpoint]['quotas']:
                                logger.info(f"  Removing {model_id} -> {endpoint} -> {quota_type}")
                                model['endpoints'][endpoint]['quotas'][quota_type] = None
                                modified = True
        
        if modified:
            save_yaml(yaml_file, data)
            logger.info(f"  ✓ Updated {yaml_file}")
    
    def _generate_csv(self):
        """Generate CSV file with valid entries"""
        valid_rows = [
            [e['model_id'], e['endpoint'], e['quota_type'], e['quota_code'], e['quota_name']]
            for e in self.entries if e.get('quota_name') != 'ERROR'
        ]
        
        write_csv(
            'metadata/quota-index.csv',
            ['model_id', 'endpoint', 'quota_type', 'quota_code', 'quota_name'],
            valid_rows
        )
        
        logger.info(f"\n✓ Generated metadata/quota-index.csv with {len(valid_rows)} valid entries")
        if self.error_entries:
            logger.info(f"✓ Cleaned up {len(self.error_entries)} ERROR entries from YAML files")


def main():
    """Main entry point"""
    generator = QuotaIndexGenerator()
    generator.run()


if __name__ == "__main__":
    main()
