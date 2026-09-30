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


def create_client(service: str, region: Optional[str] = None,
                  max_pool_connections: int = DEFAULT_MAX_POOL_CONNECTIONS):
    """Create a boto3 client for ``service`` in ``region`` with adaptive retries."""
    config = Config(
        retries={'max_attempts': 10, 'mode': 'adaptive'},
        max_pool_connections=max_pool_connections,
    )
    if region:
        return boto3.client(service, region_name=region, config=config)
    return boto3.client(service, config=config)
