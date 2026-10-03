# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate quota index CSV for validation"""

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict
import sys

from bedrock_usage_analyzer.utils.yaml_handler import load_yaml, save_yaml
from bedrock_usage_analyzer.utils.csv_handler import write_csv
from bedrock_usage_analyzer.utils.paths import list_data_files, get_writable_path, get_bundle_path
from bedrock_usage_analyzer.aws.servicequotas import check_quota, confirm_statuses, list_quota_codes, QUOTA_ERROR, QUOTA_OK, QUOTA_MISSING
from bedrock_usage_analyzer.aws.bedrock import get_regional_profile_prefixes
from bedrock_usage_analyzer.sync.quota_rules import mapping_conflict, scrub_conflicting

logger = logging.getLogger(__name__)

# 'partition' tells apart identical rows of commercial and GovCloud lists
CSV_HEADERS = ['model_id', 'endpoint', 'quota_type', 'quota_code', 'quota_name', 'partition']


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
        self._listings = {}  # region -> {code: quota} from ListServiceQuotas (None: not listed)
        self._checked_regions = set()  # regions the account can call (regions.yml)
        self._regional = set()
        self._mismatched = set()
        # Parsed fm-lists of the credentials' partition, by region (read once per run)
        self._fm_data = {}

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
        """Load the FM lists and merge endpoints from all regions of each partition

        Every partition's lists go into the index (so a GovCloud run does not drop the
        commercial rows, and vice versa), but only the credentials' partition is
        validated against Service Quotas and cleaned up.
        """
        from bedrock_usage_analyzer.sync.regions import SKIP_REGIONS, credentials_partition_or_exit
        from bedrock_usage_analyzer.utils.partition import get_partition_for_region

        self._partition = credentials_partition_or_exit()
        fm_files = []
        for fm_file in list_data_files('fm-list-*.yml'):
            filename = fm_file.name if hasattr(fm_file, 'name') else str(fm_file)
            region = filename.replace('fm-list-', '').replace('.yml', '')
            # Disrupted regions are skipped like in every refresh (their endpoints time out)
            if region not in SKIP_REGIONS:
                fm_files.append((region, fm_file))

        if not fm_files:
            logger.error("No fm-list files found")
            sys.exit(1)

        # Each model endpoint is validated in the first region listing a mapping for it, so
        # order the partition's home region first, then regions the account has enabled
        # (the user's regions.yml), and opt-in regions it may not have enabled last
        from bedrock_usage_analyzer.sync.regions import read_region_file
        from bedrock_usage_analyzer.utils.partition import PARTITION_HOME_REGIONS
        enabled = set(read_region_file(get_writable_path('regions.yml')))
        homes = set(PARTITION_HOME_REGIONS.values())
        # Only regions the account can call are validated and cleaned: an opt-in region it
        # has not enabled answers every lookup with an error (its codes are kept as they are)
        from bedrock_usage_analyzer.sync.regions import load_region_names
        from bedrock_usage_analyzer.utils.partition import filter_regions_by_partition
        # Only this partition's regions: a regions.yml written with other credentials says
        # nothing about which regions of this partition are enabled
        known = set(filter_regions_by_partition(enabled, self._partition)) or \
            set(filter_regions_by_partition(load_region_names(), self._partition))
        fm_files.sort(key=lambda item: (item[0] not in homes, item[0] not in enabled, item[0]))

        logger.info(f"Found {len(fm_files)} fm-list files")

        for region, fm_file in fm_files:
            data = load_yaml(str(fm_file)) or {}
            partition = get_partition_for_region(region)
            if partition == self._partition:
                self._fm_data[region] = data

            for model in data.get('models', []):
                key = (partition, model['model_id'])

                if key not in self.models:
                    # First time seeing this model in this partition - initialize
                    self.models[key] = {
                        'model_id': model['model_id'],
                        'partition': partition,
                        'provider': model.get('provider'),
                        'inference_types': model.get('inference_types', []),
                        'inference_profiles': model.get('inference_profiles', []),
                        'endpoints': {}
                    }

                # Merge endpoints from this region, to the dictionary that aggregates the partition
                self._merge_endpoints(key, model, region)

        # Without any regions list, every fm-list region of the partition is checked
        self._checked_regions = (known or set(self._fm_data)) | homes
        logger.info(f"Loaded {len(self.models)} unique models\n")

    def _merge_endpoints(self, key, model: Dict, region: str):
        """Merge endpoints from model into existing model entry"""
        new_endpoints = model.get('endpoints') or {}

        for endpoint_type, endpoint_data in new_endpoints.items():
            endpoint_data = endpoint_data if isinstance(endpoint_data, dict) else {}  # 'us: null'
            existing_endpoints = self.models[key]['endpoints']

            if endpoint_type not in existing_endpoints:
                # New endpoint - add it
                existing_endpoints[endpoint_type] = {
                    **endpoint_data,
                    '_source_region': region
                }
            else:
                # Endpoint exists, potentially from other regions - check if new one has quotas
                existing_quotas = existing_endpoints[endpoint_type].get('quotas', {}) or {}
                new_quotas = endpoint_data.get('quotas', {}) or {}

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
        
        for model in self.models.values():
            model_id = model['model_id']
            endpoints = model.get('endpoints', {})
            
            for endpoint_type, endpoint_data in endpoints.items():
                quotas = endpoint_data.get('quotas') or {}
                source_region = endpoint_data.get('_source_region', 'unknown')
                
                for quota_type, quota_data in quotas.items():
                    # {code: L-xxx, name: "..."} or null
                    if quota_data and isinstance(quota_data, dict):
                        quota_code = quota_data.get('code')
                        quota_name = quota_data.get('name')
                        
                        if quota_code:
                            key = (model['partition'], model_id, endpoint_type, quota_type, quota_code)
                            if key not in seen:
                                seen.add(key)
                                self.entries.append({
                                    'model_id': model_id,
                                    'endpoint': endpoint_type,
                                    'quota_type': quota_type,
                                    'quota_code': quota_code,
                                    'quota_name': quota_name,
                                    'source_region': source_region,
                                    'partition': model['partition'],
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
        # Only codes of the credentials' partition can be looked up with these credentials
        own = [e for e in self.entries if e['partition'] == self._partition]
        keys = sorted({(e['quota_code'], e['source_region']) for e in own})
        # One listing per region (a few paginated calls) instead of one call per code; the
        # cleanup reuses these listings and results
        regions = sorted((set(self._fm_data) | {region for _, region in keys}) & self._checked_regions)
        with ThreadPoolExecutor(max_workers=8) as pool:
            self._listings = dict(zip(regions, pool.map(list_quota_codes, regions)))
            cache = dict(zip(keys, pool.map(lambda key: self._lookup(*key), keys)))

        unverified = 0
        for entry in self.entries:
            if entry['partition'] != self._partition:
                # Another partition: kept in the index with its stored name, not validated,
                # and never cleaned up with these credentials
                entry['quota_name'] = entry.get('quota_name') or 'N/A'
                continue
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
        # Quota availability differs by region: a code can exist in the entry's source region
        # and be missing in another (or the reverse), so every (code, region) pair an fm-list
        # uses is checked, and a code is removed only from the regions where it is missing
        self._regional = set(get_regional_profile_prefixes())
        self._mismatched = {(e['model_id'], e['endpoint'], e['quota_type'], e['quota_code'])
                            for e in self.mismatch_entries}
        regions = sorted(set(self._fm_data) & self._checked_regions)
        pending = {(slot[3], region) for region in regions
                   for slot in self._quota_slots_in(self._fm_data[region])}
        # Codes absent from a region's listing (or in a region that could not be listed) are
        # confirmed one by one, so a code is never removed on the listing alone
        unresolved = sorted(k for k in pending - set(self._region_checks)
                            if k[0] not in (self._listings.get(k[1]) or {}))
        # (lookup through this module's name, so tests can stub it)
        confirm_statuses(unresolved, self._region_checks, lookup=lambda code, region: check_quota(code, region))
        for key in pending - set(self._region_checks):
            self._region_checks[key] = QUOTA_OK  # in the region's listing
        # Every region file is also checked for mismatches by the quota name stored with each
        # code, because the index itself samples only one source region per model endpoint
        # Mismatches (by stored name, no API call) are cleaned in every region of the partition
        for region in sorted(self._fm_data):
            self._cleanup_region_errors(region)

        # A code confirmed in any region is still a valid mapping for the index
        confirmed = {code for (code, _), status in self._region_checks.items() if status == QUOTA_OK}
        for entry in list(self.error_entries):
            if entry['quota_code'] in confirmed:
                entry['quota_name'] = entry.get('previous_name') or 'N/A'
                self.error_entries.remove(entry)

    @staticmethod
    def _quota_slots_in(data):
        """(model, endpoint, type, code) of every mapped quota in one parsed fm-list."""
        for model in data.get('models') or []:
            for endpoint, endpoint_data in (model.get('endpoints') or {}).items():
                if not isinstance(endpoint_data, dict):
                    continue
                for quota_type, quota in (endpoint_data.get('quotas') or {}).items():
                    if isinstance(quota, dict) and quota.get('code'):
                        yield (model['model_id'], endpoint, quota_type, quota['code'])

    def _lookup(self, code: str, region: str):
        """(status, quota) from the region's listing, else from GetServiceQuota (recorded for the cleanup)."""
        if region not in self._checked_regions:
            return (QUOTA_ERROR, None)  # not enabled for the account: kept, not looked up
        listed = (self._listings.get(region) or {}).get(code)
        result = (QUOTA_OK, listed) if listed else check_quota(code, region)
        self._region_checks[(code, region)] = result[0]
        return result

    def _missing_in(self, code: str, region: str) -> bool:
        if region not in self._checked_regions:
            return False  # not enabled for the account: cannot be verified, kept
        key = (code, region)
        if key not in self._region_checks:
            self._region_checks[key] = check_quota(code, region)[0]
        return self._region_checks[key] == QUOTA_MISSING

    def _cleanup_region_errors(self, region: str):
        """Null out codes missing in this region or contradicting their model/endpoint (user copy, else bundled)"""
        regional, mismatched = self._regional, self._mismatched
        data = self._fm_data[region]  # only the credentials' partition is cleaned

        modified = False
        reasons = set()
        for model in data.get('models', []):
            for endpoint, endpoint_data in (model.get('endpoints') or {}).items():
                quotas = (endpoint_data or {}).get('quotas') or {}
                # Contradicting this model/endpoint by the stored name (the same rule the
                # analyzer and fm-quotas apply), then codes the index flagged or the region lacks
                removed = [(t, q, 'mismatch') for t, q, _ in
                           scrub_conflicting(model['model_id'], endpoint, quotas, regional)]
                for quota_type, quota in quotas.items():
                    if not isinstance(quota, dict):
                        continue
                    code = quota.get('code')
                    if (model['model_id'], endpoint, quota_type, code) in mismatched:
                        removed.append((quota_type, quota, 'mismatch'))
                    elif code and self._missing_in(code, region):
                        removed.append((quota_type, quota, 'missing'))
                for quota_type, quota, reason in removed:
                    reasons.add(reason)
                    logger.info(f"  Removing {model['model_id']} -> {endpoint} -> {quota_type} "
                                f"({quota.get('code')}) in {region}: {reason}")
                    quotas[quota_type] = None
                    modified = True

        if not modified:
            return
        user_file = get_writable_path(f'fm-list-{region}.yml')
        # Only an existing user copy is rewritten. Creating one from a bundled list would hide
        # every later bundled update for that region; the analyzer applies the same checks
        # when it reads quotas, so bundled lists are corrected by maintainers (--update-bundle).
        if user_file.exists():
            save_yaml(str(user_file), data)
            logger.info(f"  ✓ Updated {user_file}")
        bundle_path = get_bundle_path() if self.update_bundle else None
        if bundle_path:
            bundle_file = bundle_path / f'fm-list-{region}.yml'
            save_yaml(str(bundle_file), data)
            logger.info(f"  ✓ Updated {bundle_file} (bundled)")
        elif not user_file.exists():
            # The analyzer re-applies the mismatch checks; a missing code is looked up and
            # reported as missing, then the report shows usage without that limit
            effect = 'the analyzer skips mismatched codes' if reasons == {'mismatch'} else \
                'the analyzer skips mismatched codes and shows usage without the missing limits'
            logger.info(f"  (bundled list for {region} left unchanged ({effect}); a maintainer fixes "
                        f"it with --update-bundle)")

    def _generate_csv(self):
        """Generate CSV file with valid entries"""
        valid_rows = [
            [e['model_id'], e['endpoint'], e['quota_type'], e['quota_code'], e['quota_name'], e['partition']]
            for e in self.entries if e.get('quota_name') not in ('ERROR', 'MISMATCH')
        ]
        
        output_file = get_writable_path('quota-index.csv')
        write_csv(
            str(output_file),
            CSV_HEADERS,
            valid_rows
        )
        logger.info(f"\n✓ Generated {output_file} with {len(valid_rows)} valid entries")
        
        if getattr(self, 'update_bundle', False):
            bundle_path = get_bundle_path()
            if bundle_path:
                bundle_file = bundle_path / 'quota-index.csv'
                write_csv(
                    str(bundle_file),
                    CSV_HEADERS,
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
