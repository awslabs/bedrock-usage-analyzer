# AWS GovCloud Setup Guide

## Why GovCloud Regions Don't Appear

If you're not seeing GovCloud regions (like `us-gov-east-1`, `us-gov-west-1`) in the available regions list when running `bedrock-usage-analyzer analyze`, it's because:

1. **Separate Credentials Required**: GovCloud requires completely separate AWS credentials from standard AWS
2. **Regions Not Discovered**: The regions list needs to be refreshed with GovCloud credentials
3. **Security Clearance**: GovCloud access requires appropriate US government security clearance

## Quick Fix

To get GovCloud regions to appear in the regions list:

### Step 1: Configure GovCloud Credentials

You need separate GovCloud AWS credentials. Add them to your `~/.aws/credentials` file:

```ini
[govcloud]
aws_access_key_id = YOUR_GOVCLOUD_ACCESS_KEY
aws_secret_access_key = YOUR_GOVCLOUD_SECRET_KEY
region = us-gov-east-1
```

### Step 2: Refresh Regions with GovCloud Credentials

Run the regions refresh command with your GovCloud profile:

```bash
AWS_PROFILE=govcloud bedrock-usage-analyzer refresh regions
```

This will discover and add GovCloud regions to your regions metadata.

### Step 3: Verify GovCloud Regions

After refreshing, run the analyze command again:

```bash
bedrock-usage-analyzer analyze
```

You should now see GovCloud regions marked with 🏛️ indicators in the regions list.

## Alternative: Mixed Credentials Approach

If you have both standard AWS and GovCloud credentials, you can discover regions from both partitions:

### Step 1: Discover Standard Regions
```bash
# Use your standard AWS credentials (default profile)
bedrock-usage-analyzer refresh regions
```

### Step 2: Discover GovCloud Regions
```bash
# Use your GovCloud credentials
AWS_PROFILE=govcloud bedrock-usage-analyzer refresh regions
```

The tool will merge regions from both discoveries.

## Troubleshooting

### "Could not discover GovCloud regions" Error

This is normal if you don't have GovCloud credentials configured. The tool will still work with standard AWS regions.

### "No regions found" Error

This means the tool couldn't discover any regions. Check:
1. Your AWS credentials are configured correctly
2. You have network connectivity to AWS
3. Your credentials have the necessary permissions

### GovCloud Regions Still Not Showing

1. **Verify Credentials**: Test your GovCloud credentials:
   ```bash
   AWS_PROFILE=govcloud aws sts get-caller-identity
   ```

2. **Check Permissions**: Ensure your GovCloud credentials have permissions for:
   - `account:ListRegions`
   - `bedrock:ListFoundationModels`
   - `cloudwatch:GetMetricStatistics`
   - `servicequotas:GetServiceQuota`

3. **Manual Region Addition**: If automatic discovery fails, you can manually add GovCloud regions to your user data directory's `regions.yml`:
   ```yaml
   regions:
   - name: us-east-1
     type: standard
     display_name: US East (N. Virginia)
     partition: aws
   - name: us-gov-east-1
     type: govcloud
     display_name: AWS GovCloud (US-East)
     partition: aws-us-gov
   - name: us-gov-west-1
     type: govcloud
     display_name: AWS GovCloud (US-West)
     partition: aws-us-gov
   ```

## Understanding GovCloud

### What is AWS GovCloud?

AWS GovCloud (US) is an isolated AWS region designed for US government agencies and contractors to host sensitive data and regulated workloads in the cloud.

### Key Differences

1. **Separate Partition**: GovCloud is in the `aws-us-gov` partition, separate from standard AWS (`aws`)
2. **Different Endpoints**: Services use different endpoints (e.g., `bedrock.us-gov-east-1.amazonaws.com`)
3. **Limited Services**: Not all AWS services are available in GovCloud
4. **Compliance**: Designed for FedRAMP, ITAR, and other compliance requirements

### GovCloud Regions

Currently available GovCloud regions:
- `us-gov-east-1` - AWS GovCloud (US-East)
- `us-gov-west-1` - AWS GovCloud (US-West)

## Security Notes

- **Credentials Separation**: Never use standard AWS credentials for GovCloud or vice versa
- **Data Isolation**: Data in GovCloud stays within the GovCloud partition
- **Access Control**: GovCloud access requires appropriate security clearance
- **Compliance**: Ensure your use complies with your organization's security policies

## Getting GovCloud Access

To get GovCloud access:

1. **Eligibility**: Must be a US government agency, contractor, or have qualifying use case
2. **Application**: Apply through AWS GovCloud sales team
3. **Verification**: AWS will verify eligibility and compliance requirements
4. **Setup**: Once approved, you'll receive separate GovCloud credentials

For more information, visit: https://aws.amazon.com/govcloud-us/

## Support

If you continue having issues with GovCloud regions:

1. **Check AWS Documentation**: https://docs.aws.amazon.com/govcloud-us/
2. **Verify Credentials**: Ensure your GovCloud credentials are valid
3. **Contact Support**: Reach out to AWS GovCloud support for access issues
4. **Review Logs**: Check the tool's log output for specific error messages