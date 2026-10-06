# Testing Guide - GovCloud Support

## Quick Test Steps

### 1. Verify Code Compilation
All code compiles successfully:
```bash
✅ partition.py
✅ bedrock.py
✅ sts.py
✅ analyzer.py
✅ output_generator.py
```

### 2. Install Latest Version
```bash
# From repository root
pipx install -e . --force
```

### 3. Test in GovCloud

#### Set up GovCloud credentials:
```bash
# Option 1: Using AWS Profile
export AWS_PROFILE=my-govcloud-profile

# Option 2: Direct credentials
export AWS_ACCESS_KEY_ID=AKIAxxxxxxxxx
export AWS_SECRET_ACCESS_KEY=xxxxxxxxxxxxxxxx
export AWS_REGION=us-gov-west-1
```

#### Run the analyzer:
```bash
# Interactive mode
bedrock-usage-analyzer analyze

# Non-interactive mode (if you know the model ID)
bedrock-usage-analyzer analyze \
  --region us-gov-west-1 \
  --model-id amazon.nova-micro-v1:0 \
  --granularity 1min \
  --yes
```

### 4. Verify Output

#### Check JSON Output:
```bash
# Look for GovCloud ARN format
cat results/*.json | grep "arn:aws-us-gov"

# Should see ARNs like:
# "arn:aws-us-gov:bedrock:us-gov-west-1::foundation-model/..."
```

#### Check HTML Output:
```bash
# Open in browser
open results/*.html  # macOS
# or
xdg-open results/*.html  # Linux
```

Verify in HTML:
- ✅ Console URLs point to `console.amazonaws-us-gov.com`
- ✅ Quota links work and point to GovCloud console
- ✅ No broken links to commercial AWS console

### 5. Test in Commercial AWS (Regression Test)

```bash
# Switch to commercial AWS credentials
export AWS_PROFILE=my-commercial-profile
export AWS_REGION=us-west-2

# Run analyzer
bedrock-usage-analyzer analyze

# Verify output uses commercial ARNs and URLs
cat results/*.json | grep "arn:aws:bedrock"
# Should see: "arn:aws:bedrock:us-west-2::..."
```

## Expected Behavior

### GovCloud Environment:
- **Partition Detection**: `aws-us-gov`
- **ARN Format**: `arn:aws-us-gov:bedrock:us-gov-west-1::...`
- **Console Domain**: `console.amazonaws-us-gov.com`
- **Quota URLs**: `https://us-gov-west-1.console.amazonaws-us-gov.com/servicequotas/...`

### Commercial AWS:
- **Partition Detection**: `aws`
- **ARN Format**: `arn:aws:bedrock:us-west-2::...`
- **Console Domain**: `console.aws.amazon.com`
- **Quota URLs**: `https://us-west-2.console.aws.amazon.com/servicequotas/...`

## Troubleshooting

### Issue: "No module named 'bedrock_usage_analyzer'"
**Solution:** Reinstall the package:
```bash
pipx install -e . --force
```

### Issue: "No module named 'boto3'"
**Solution:** The package dependencies should be installed automatically. If not:
```bash
pip install boto3
```

### Issue: "AccessDenied" in GovCloud
**Solution:** Verify:
1. You have access to Bedrock in GovCloud
2. Your IAM permissions include required actions
3. Bedrock is enabled in your GovCloud account
4. The model you're trying to analyze is available in GovCloud

### Issue: Console URLs not working
**Solution:**
- Verify you have access to AWS Console in GovCloud
- Check network connectivity to GovCloud endpoints
- For ISO partitions, additional network configuration may be required

## Testing Checklist

Before considering the implementation complete, verify:

- [ ] Code compiles without errors
- [ ] Package installs successfully with pipx
- [ ] GovCloud partition is detected correctly
- [ ] ARNs use correct partition format
- [ ] Console URLs point to correct domain
- [ ] Quota links work in browser
- [ ] Generated reports display correctly
- [ ] Commercial AWS still works (regression test)
- [ ] No errors or warnings during execution

## Sample Output

### Successful GovCloud Detection:
```
Detected AWS partition: aws-us-gov
Current partition: aws-us-gov
Console domain: console.amazonaws-us-gov.com
Sample ARN: arn:aws-us-gov:bedrock:us-gov-west-1::foundation-model/test-model
```

### Successful Commercial Detection:
```
Detected AWS partition: aws
Current partition: aws
Console domain: console.aws.amazon.com
Sample ARN: arn:aws:bedrock:us-west-2::foundation-model/test-model
```

## Automated Testing

Run the test suite:
```bash
python3 tests/test_partition.py
```

Expected output:
```
============================================================
Partition Support Tests
============================================================

=== Testing ARN Construction ===
[Test results]

=== Testing Region Detection ===
  ✓ us-gov-west-1 detected as GovCloud
  ✓ us-gov-east-1 detected as GovCloud
  ✓ cn-north-1 detected as China
  [etc.]

=== Testing Current Partition ===
  Current partition: aws-us-gov (or aws)
  Console domain: console.amazonaws-us-gov.com (or console.aws.amazon.com)
  [etc.]

============================================================
All partition support tests passed!
```

## Next Steps After Testing

1. If all tests pass in GovCloud → Ready for production use
2. If any issues → Review error messages and check GOVCLOUD_SUPPORT.md
3. Consider creating a git commit with the changes
4. Update version number if releasing
5. Test with various Bedrock models available in your partition
