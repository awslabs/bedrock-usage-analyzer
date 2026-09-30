# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""AWS regions management across partitions (commercial, GovCloud, China)"""

import logging
import os
import sys
from typing import Iterable, List, Optional

from bedrock_usage_analyzer.aws.client_factory import create_client
from bedrock_usage_analyzer.utils.partition import (
    PARTITION_HOME_REGIONS,
    detect_credentials_partition,
    filter_regions_by_partition,
    get_partition_display_name,
    get_partition_for_region,
    partition_regions,
    region_hint,
)
from bedrock_usage_analyzer.utils.paths import get_data_path
from bedrock_usage_analyzer.utils.yaml_handler import load_yaml

logger = logging.getLogger(__name__)

# Regions to skip due to region disruption
SKIP_REGIONS = {'me-south-1', 'me-central-1'}


def _region_name(entry) -> Optional[str]:
    """Accept both plain names and {'name': ...} entries in regions.yml."""
    if isinstance(entry, dict):
        return entry.get('name')
    if isinstance(entry, str):
        return entry
    return None


def normalize_region_names(entries: Iterable) -> List[str]:
    """Return a sorted, de-duplicated list of region names from regions.yml entries."""
    names = {_region_name(e) for e in (entries or [])}
    return sorted(n for n in names if n)


def read_region_file(path) -> List[str]:
    """Region names from one regions.yml file ([] if it does not exist)."""
    if not path or not os.path.exists(str(path)):
        return []
    return normalize_region_names((load_yaml(str(path)) or {}).get('regions', []))


def load_region_names() -> List[str]:
    """Load region names: the user's regions.yml, plus bundled regions of partitions it lacks.

    A user file written before GovCloud was bundled lists only commercial
    regions; it still wins for commercial, but GovCloud comes from the bundle.
    """
    from bedrock_usage_analyzer.utils.paths import load_bundled_yaml
    names = read_region_file(get_data_path('regions.yml'))
    bundled = normalize_region_names((load_bundled_yaml('regions.yml') or {}).get('regions', []))
    present = {get_partition_for_region(r) for r in names}
    return sorted(set(names) | {r for r in bundled if get_partition_for_region(r) not in present})


def credentials_partition_or_exit(region: Optional[str] = None) -> str:
    """Partition of the current credentials; exit with advice when it cannot be read.

    Commands that act on "the regions your credentials can call" must not guess:
    guessing processes (or saves) regions of the wrong partition.
    """
    partition = detect_credentials_partition(region or region_hint())
    if partition is None:
        logger.error("Could not read the caller identity; check your AWS credentials "
                     "(aws sts get-caller-identity); for GovCloud set AWS_REGION, e.g. AWS_REGION=us-gov-west-1.")
        sys.exit(1)
    return partition


def regions_for_credentials(regions: Iterable[str], region: Optional[str] = None):
    """Filter regions to the partition of the current credentials.

    Returns (regions, partition). Exits when the partition cannot be detected.
    """
    partition = credentials_partition_or_exit(region)
    filtered = filter_regions_by_partition(regions, partition)
    if not filtered:
        logger.warning(f"No {get_partition_display_name(partition)} regions in regions.yml; "
                       f"run: bua refresh regions")
    return filtered, partition


def _home_region(partition: str, hint: Optional[str]) -> Optional[str]:
    """A region in ``partition`` to pin regional API calls to."""
    if hint and get_partition_for_region(hint) == partition:
        return hint
    candidates = partition_regions(partition)
    if PARTITION_HOME_REGIONS.get(partition) in candidates:
        return PARTITION_HOME_REGIONS[partition]
    return candidates[0] if candidates else None


def _fetch_via_account_api(region: Optional[str]) -> List[str]:
    client = create_client('account', region)
    regions = []
    paginator = client.get_paginator('list_regions')
    for page in paginator.paginate(RegionOptStatusContains=['ENABLED', 'ENABLED_BY_DEFAULT']):
        regions.extend(r['RegionName'] for r in page.get('Regions', []))
    return regions


def _fetch_via_ec2(region: Optional[str]) -> List[str]:
    # DescribeRegions without AllRegions returns only regions enabled for the account
    response = create_client('ec2', region).describe_regions()
    return [r['RegionName'] for r in response.get('Regions', [])]


def fetch_enabled_regions(partition: Optional[str] = None, region: Optional[str] = None) -> List[str]:
    """Fetch the regions enabled for the account in the credentials' partition.

    Tries the Account Management API first, then EC2 DescribeRegions, then
    botocore's static region list for Bedrock in that partition.
    """
    hint = region or region_hint()
    # Without working credentials the static fallback below would silently
    # replace the list with every region, including ones never enabled
    partition = partition or credentials_partition_or_exit(hint)
    home = _home_region(partition, hint)

    errors = []
    for name, fetch in (('account:ListRegions', _fetch_via_account_api),
                        ('ec2:DescribeRegions', _fetch_via_ec2)):
        try:
            regions = filter_regions_by_partition(fetch(home), partition)
            if regions:
                return sorted(set(regions))
        except Exception as e:
            errors.append(f"{name}: {e}")
            logger.debug(f"{name} failed: {e}")

    import boto3
    static = boto3.session.Session().get_available_regions('bedrock', partition)
    if static:
        for err in errors:
            logger.warning(f"  Could not list enabled regions via {err}")
        logger.warning(f"  Using the {len(static)} Bedrock regions botocore knows for {partition}")
        return sorted(static)

    for err in errors:
        logger.error(f"Error fetching regions via {err}")
    sys.exit(1)


def merge_regions(existing: Iterable, fresh: Iterable[str], partition: str) -> List[str]:
    """Replace the regions of ``partition`` in ``existing`` with ``fresh``.

    Regions of other partitions are kept, so refreshing with commercial
    credentials does not drop the GovCloud regions (and vice versa).
    """
    kept = [r for r in normalize_region_names(existing) if get_partition_for_region(r) != partition]
    return sorted(set(kept) | set(fresh))


def discover_regions():
    """Return (partition, regions) enabled for the current credentials, minus SKIP_REGIONS."""
    hint = region_hint()
    partition = credentials_partition_or_exit(hint)
    logger.info(f"Fetching enabled regions ({get_partition_display_name(partition)})...")

    regions = fetch_enabled_regions(partition, hint)
    skipped = sorted(set(regions) & SKIP_REGIONS)
    regions = [r for r in regions if r not in SKIP_REGIONS]
    if skipped:
        logger.info(f"Skipping regions: {', '.join(skipped)}")

    if not regions:
        logger.error("No regions found")
        sys.exit(1)

    logger.info(f"Found {len(regions)} enabled regions")
    return partition, regions


def refresh_regions(existing: Optional[Iterable] = None, discovered=None):
    """Refresh the regions list for the partition of the current credentials.

    Args:
        existing: Current regions.yml entries; regions of other partitions are kept.
        discovered: Optional (partition, regions) from discover_regions(), to reuse one lookup.

    Returns:
        dict: Regions data {'regions': [...]}
    """
    partition, regions = discovered or discover_regions()
    merged = merge_regions(existing or [], regions, partition)
    other = len(merged) - len(regions)
    if other:
        logger.info(f"Kept {other} region(s) from other partitions")
    return {'regions': merged}


def main():
    """Main entry point"""
    refresh_regions()


if __name__ == "__main__":
    main()
