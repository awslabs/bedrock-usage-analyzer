# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Secure CSV file operations using defusedcsv"""

from defusedcsv import csv


def write_csv(filepath, headers, rows):
    """Write data to CSV file securely
    
    Args:
        filepath: Path to CSV file
        headers: List of column headers
        rows: List of row data (list of lists)
    """
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def read_csv(filepath):
    """Read CSV file securely
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        tuple: (headers, rows)
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        headers = next(reader)
        rows = list(reader)
        return headers, rows
