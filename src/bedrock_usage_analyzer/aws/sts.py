# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""AWS STS (Security Token Service) operations"""

import boto3
from bedrock_usage_analyzer.utils.partition import get_account_id as _get_account_id, get_partition


def get_account_id():
    """Get current AWS account ID

    Returns:
        str: AWS account ID

    Raises:
        Exception: If unable to get account ID
    """
    # Use cached value from partition module to avoid duplicate API calls
    account_id = _get_account_id()
    if account_id is None:
        # Fallback to direct call if not cached
        sts = boto3.client('sts')
        account_id = sts.get_caller_identity()['Account']
    return account_id
