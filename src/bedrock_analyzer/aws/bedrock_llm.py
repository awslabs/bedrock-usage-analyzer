# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bedrock LLM invocation for intelligent quota mapping"""

import boto3
import sys
from typing import Optional, Dict, List

from bedrock_analyzer.aws.bedrock import get_endpoint_descriptions


def extract_common_name(region: str, model_id: str, fm_model_id: str) -> Optional[str]:
    """Extract common model name using LLM with tool call
    This method is used in quotas mapping (mapping FM usage metric e.g. RPM to the correct quota in AWS Service Quotas)
    This instructs LLM to extract a common name of an FM. For example, 'nova-lite' becomes 'nova'
    The technique is used to form keyword search to narrow down the list of FMs to be fed into the LLM call for the actual mapping, while avoiding false negatives of being too specific
    
    Args:
        region: AWS region for Bedrock
        model_id: Model ID to use for extraction
        fm_model_id: Foundation model ID to extract name from
        
    Returns:
        Common name or None
    """
    client = boto3.client('bedrock-runtime', region_name=region)
    
    # Use tool to enforce JSON format
    tool_config = {
        'tools': [{
            'toolSpec': {
                'name': 'report_common_name',
                'description': 'Report the base model family name',
                'inputSchema': {
                    'json': {
                        'type': 'object',
                        'properties': {
                            'common_name': {
                                'type': 'string',
                                'description': 'The base model family name (e.g., "nova", "claude")'
                            }
                        },
                        'required': ['common_name']
                    }
                }
            }
        }],
        'toolChoice': {'tool': {'name': 'report_common_name'}}
    }
    
    prompt = f"""Extract the base model family name from this model ID: {fm_model_id}

Examples:
- "amazon.nova-lite-v1:0" → "nova"
- "anthropic.claude-3-5-sonnet-20241022-v2:0" → "claude"
- "us.anthropic.claude-haiku-4-5-20251001-v1:0" → "claude"

Use the report_common_name tool to provide ONLY the base family name."""

    try:
        response = client.converse(
            modelId=model_id,
            messages=[{'role': 'user', 'content': [{'text': prompt}]}],
            toolConfig=tool_config,
            inferenceConfig={'maxTokens': 50, 'temperature': 0}
        )
        
        content = response['output']['message']['content']
        for block in content:
            if 'toolUse' in block:
                common_name = block['toolUse']['input'].get('common_name', '').strip().lower()
                return common_name if common_name else None
        
        return None
        
    except Exception as e:
        print(f"Error extracting common name: {e}", file=sys.stderr)
        return None


def extract_quota_codes(region: str, model_id: str, fm_model_id: str, 
                       endpoint_type: str, matching_quotas: List[Dict]) -> Optional[Dict]:
    """Extract quota codes for all matching quotas in one LLM call
    This method uses LLM's intelligence to map the right quotas in AWS Service Quotas to a specific metric (e.g. RPM) of a given FM.
    
    Args:
        region: AWS region for Bedrock
        model_id: Model ID to use for extraction
        fm_model_id: Foundation model ID being mapped
        endpoint_type: Endpoint type (base/us/eu/global/etc)
        matching_quotas: List of matching quota dicts with 'name', 'code', and 'value'
        
    Returns:
        Dict with tpm/rpm/tpd/concurrent, each containing {code, name} or None
    """
    client = boto3.client('bedrock-runtime', region_name=region)
    
    tool_config = {
        'tools': [{
            'toolSpec': {
                'name': 'report_quota_mapping',
                'description': 'Report the quota codes for TPM, RPM, TPD, and Concurrent Requests',
                'inputSchema': {
                    'json': {
                        'type': 'object',
                        'properties': {
                            'tpm_quota_code': {
                                'type': ['string', 'null'],
                                'description': 'Quota code for Tokens Per Minute (TPM), or null if not found'
                            },
                            'rpm_quota_code': {
                                'type': ['string', 'null'],
                                'description': 'Quota code for Requests Per Minute (RPM), or null if not found'
                            },
                            'tpd_quota_code': {
                                'type': ['string', 'null'],
                                'description': 'Quota code for Tokens Per Day (TPD), or null if not found'
                            },
                            'concurrent_requests_quota_code': {
                                'type': ['string', 'null'],
                                'description': 'Quota code for Concurrent Requests, or null if not found'
                            }
                        },
                        'required': ['tpm_quota_code', 'rpm_quota_code', 'tpd_quota_code', 'concurrent_requests_quota_code']
                    }
                }
            }
        }],
        'toolChoice': {'tool': {'name': 'report_quota_mapping'}}
    }
    
    quotas_text = "\n".join([f"- {q['name']} (code: {q['code']})" for q in matching_quotas])
    
    endpoint_descriptions = get_endpoint_descriptions()
    endpoint_desc = endpoint_descriptions.get(endpoint_type, endpoint_type)
    
    prompt = f"""For the Bedrock model "{fm_model_id}" with {endpoint_desc} endpoint, identify which quota codes correspond to:
- TPM (Tokens Per Minute)
- RPM (Requests Per Minute)  
- TPD (Tokens Per Day)
- Concurrent Requests (if available, some models use this instead of or in addition to RPM)

Available quotas:
{quotas_text}

CRITICAL MATCHING RULES for model ID "{fm_model_id}":

1. EXACT SUBSTRING MATCH - The model variant/generation in the model ID must appear in the quota name:
   - "claude-3-haiku" → must find "Claude 3 Haiku" in quota (NOT "Haiku 4.5")
   - "claude-haiku-4-5" → must find "Haiku 4.5" in quota (NOT "Claude 3 Haiku")
   - "claude-3-5-sonnet" → must find "3.5 Sonnet" in quota (NOT "3 Sonnet" or "3.7 Sonnet")
   - "nova-lite" → must find "Nova Lite" in quota (NOT "Nova Sonic" or "Nova Pro")
   - "llama3-2" → must find "Llama 3.2" in quota (NOT "Llama 3.1")

2. VERSION SUFFIX - Match v1:0 to V1, v2:0 to V2 in quota names

3. REJECT PARTIAL MATCHES - If generation/variant numbers don't match exactly, return null for that metric

Some models may only have concurrent requests or RPM. Return null if no exact match found.

Use the report_quota_mapping tool to provide the quota codes. If a quota type is not found, use null."""
    
    try:
        response = client.converse(
            modelId=model_id,
            messages=[{'role': 'user', 'content': [{'text': prompt}]}],
            toolConfig=tool_config,
            inferenceConfig={'maxTokens': 500, 'temperature': 0}
        )
        
        content = response['output']['message']['content']
        for block in content:
            if 'toolUse' in block:
                tool_input = block['toolUse']['input']
                
                # Build quota lookup map
                quota_map = {q['code']: q['name'] for q in matching_quotas}
                
                # Return code + name for each quota type
                result = {}
                for metric, code_key in [('tpm', 'tpm_quota_code'), ('rpm', 'rpm_quota_code'), 
                                         ('tpd', 'tpd_quota_code'), ('concurrent', 'concurrent_requests_quota_code')]:
                    code = tool_input.get(code_key)
                    if code:
                        result[metric] = {
                            'code': code,
                            'name': quota_map.get(code, 'Unknown')
                        }
                    else:
                        result[metric] = None
                
                return result
        
        return None
        
    except Exception as e:
        print(f"Error extracting quota codes: {e}", file=sys.stderr)
        return None
