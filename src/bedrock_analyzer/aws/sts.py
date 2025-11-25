# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""AWS STS (Security Token Service) operations"""

import boto3


def get_account_id():
    """Get current AWS account ID
    
    Returns:
        str: AWS account ID
        
    Raises:
        Exception: If unable to get account ID
    """
    sts = boto3.client('sts')
    return sts.get_caller_identity()['Account']
