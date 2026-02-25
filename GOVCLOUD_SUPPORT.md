# AWS GovCloud Support

This document describes the AWS GovCloud (and multi-partition) support added to the Bedrock Usage Analyzer.

## Overview

The tool now automatically detects which AWS partition it's running in and adapts ARN construction, console URLs, and other partition-specific behaviors accordingly.

## Supported Partitions

- **AWS Commercial** (`aws`) - Standard AWS regions like us-west-2, eu-west-1
- **AWS GovCloud** (`aws-us-gov`) - GovCloud regions like us-gov-west-1, us-gov-east-1
- **AWS China** (`aws-cn`) - China regions like cn-north-1, cn-northwest-1
- **AWS ISO** (`aws-iso`) - C2S regions
- **AWS ISO-B** (`aws-iso-b`) - SC2S regions

## Changes Made

### 1. New Partition Utility Module

**File:** `src/bedrock_usage_analyzer/utils/partition.py`

This new module provides partition-aware utilities:

- `get_partition()` - Detects current AWS partition from STS GetCallerIdentity
- `get_account_id()` - Returns cached account ID
- `build_arn()` - Constructs ARNs with correct partition
- `get_console_domain()` - Returns correct console domain for partition
- `get_service_quota_url()` - Builds partition-aware console URLs
- `is_govcloud_region()` - Checks if region is GovCloud
- `is_china_region()` - Checks if region is China

### 2. Updated Files

#### `src/bedrock_usage_analyzer/aws/bedrock.py`
- **Line 331**: Changed hardcoded `arn:aws:bedrock:...` to use `build_arn()` for correct partition

#### `src/bedrock_usage_analyzer/aws/sts.py`
- Updated to use cached account ID from partition module to avoid duplicate API calls

#### `src/bedrock_usage_analyzer/core/analyzer.py`
- **Line 112**: Changed hardcoded console URL to use `get_service_quota_url()` function

#### `src/bedrock_usage_analyzer/core/output_generator.py`
- **Line 66**: Updated service quotas disclaimer URL to use correct console domain
- Template now receives `console_domain` variable for dynamic URLs

#### `src/bedrock_usage_analyzer/templates/report.html`
- **Line 79**: Changed hardcoded console URL to use template variable

## How It Works

1. **Partition Detection**: On first AWS API call (STS GetCallerIdentity), the tool detects the partition from the ARN structure and caches it
2. **ARN Construction**: All ARNs are built using the detected partition (e.g., `arn:aws-us-gov:bedrock:...` for GovCloud)
3. **Console URLs**: All console URLs use the correct domain:
   - Commercial: `console.aws.amazon.com`
   - GovCloud: `console.amazonaws-us-gov.com`
   - China: `console.amazonaws.cn`

## Testing in GovCloud

To test with GovCloud credentials:

```bash
# Set up GovCloud credentials
export AWS_PROFILE=my-govcloud-profile
# or
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_REGION=us-gov-west-1

# Run the analyzer
bedrock-usage-analyzer analyze

# Or use CLI arguments for non-interactive mode
bedrock-usage-analyzer analyze \
  --region us-gov-west-1 \
  --model-id amazon.nova-micro-v1:0 \
  --granularity 1min \
  --yes
```

## Verification

You can verify the partition is being detected correctly by checking the generated output:

1. **JSON Output**: Check `results/<model>-<timestamp>.json` - all ARNs should use correct partition
2. **HTML Output**: Check `results/<model>-<timestamp>.html` - console URLs should point to correct domain
3. **Logs**: The tool will log detected partition on first run

## Example ARN Formats

### Commercial AWS
```
arn:aws:bedrock:us-west-2::foundation-model/amazon.nova-premier-v1:0
```

### AWS GovCloud
```
arn:aws-us-gov:bedrock:us-gov-west-1::foundation-model/amazon.nova-premier-v1:0
```

### AWS China
```
arn:aws-cn:bedrock:cn-north-1::foundation-model/amazon.nova-premier-v1:0
```

## Backwards Compatibility

All changes are backwards compatible:
- Existing functionality unchanged for commercial AWS users
- No breaking changes to API or command-line interface
- Automatic partition detection requires no configuration

## Known Limitations

1. **Model Availability**: Not all Bedrock models are available in all partitions. Check AWS documentation for model availability in your partition.
2. **Service Quotas**: Quota codes (L-codes) may differ between partitions.
3. **Cross-Partition**: The tool does not support analyzing resources across different partitions in a single run.

## Troubleshooting

### "No such resource" errors
- Verify the model is available in your partition and region
- Check that Bedrock is enabled in your GovCloud/China account

### Console URLs not working
- Verify you have access to the AWS Console in your partition
- For ISO partitions, console access may require additional network configuration

### Permission errors
- Ensure your IAM policy allows the required actions
- GovCloud and China may have additional compliance requirements

## Support

For issues specific to GovCloud/multi-partition support, please file an issue on GitHub with:
- The partition you're using (commercial/GovCloud/China/ISO)
- The region you're using
- Relevant error messages
- Generated ARNs (check logs or output files)
