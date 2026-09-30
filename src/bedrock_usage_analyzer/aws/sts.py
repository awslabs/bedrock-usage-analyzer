# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""AWS STS (Security Token Service) operations"""

from typing import Optional

from bedrock_usage_analyzer.utils.partition import get_caller_identity


def get_account_id(region: Optional[str] = None) -> str:
    """Get current AWS account ID (cached per STS region).

    Args:
        region: Region whose STS endpoint to use; required for GovCloud or
            China credentials when no region is configured.

    Raises:
        Exception: If unable to get account ID
    """
    return get_caller_identity(region)['Account']
