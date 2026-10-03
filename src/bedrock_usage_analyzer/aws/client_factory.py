# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single place where the tool creates boto3 clients.

botocore resolves the right endpoint for every partition (commercial,
GovCloud, China, ...) from the region name, so no endpoint URL is hardcoded
here. That also keeps AWS_ENDPOINT_URL_*, FIPS and VPC endpoint settings from
the user's AWS config working.
"""

from typing import Optional

import boto3
from botocore.config import Config

# CloudWatch fetches run in a thread pool; the pool must be at least as large as
# the number of workers or urllib3 discards connections ("Connection pool is full").
DEFAULT_MAX_POOL_CONNECTIONS = 50

# Bulk data calls (metric fetches, quota lookups, LLM mapping) are throttled under load,
# so they retry longer with client-side rate adaptation. Everything else, including the
# Bedrock control-plane calls made while the user answers prompts, uses the standard
# policy with up to 4 attempts (max_attempts counts retries) and a 10 s connect timeout
# per attempt, so a bad network fails in under a minute rather than many.
_BULK_SERVICES = {'cloudwatch', 'service-quotas', 'bedrock-runtime'}


def create_client(service: str, region: Optional[str] = None,
                  max_pool_connections: int = DEFAULT_MAX_POOL_CONNECTIONS, probe: bool = False):
    """Create a boto3 client for ``service`` in ``region``.

    ``probe=True`` is for best-effort checks (e.g. asking another partition's STS whose
    credentials these are): one attempt and short timeouts, so a blocked endpoint
    cannot delay the real error message.
    """
    if probe:
        config = Config(retries={'total_max_attempts': 1, 'mode': 'standard'},
                        connect_timeout=3, read_timeout=5)
    else:
        if service in _BULK_SERVICES:
            retries = {'max_attempts': 8, 'mode': 'adaptive'}
        else:
            retries = {'max_attempts': 3, 'mode': 'standard'}
        config = Config(retries=retries, max_pool_connections=max_pool_connections, connect_timeout=10)
    if region:
        return boto3.client(service, region_name=region, config=config)
    return boto3.client(service, config=config)
