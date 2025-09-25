# Database Encryption at Rest - Azure Configuration

## Overview
Microsoft Partner-Level implementation of database encryption following Azure Security Best Practices.

## Azure Storage Account Encryption
- **Service**: Azure Storage Account `stcultivateml`
- **Encryption**: 256-bit AES encryption (enabled by default)
- **Key Management**: Microsoft-managed keys (MMK)
- **Status**: ✅ ACTIVE

### Encryption Details
```json
{
  "storageAccount": "stcultivateml",
  "encryption": {
    "services": {
      "blob": {
        "enabled": true,
        "keyType": "Microsoft-managed"
      },
      "file": {
        "enabled": true,
        "keyType": "Microsoft-managed"
      }
    },
    "keySource": "Microsoft.Storage",
    "requireInfrastructureEncryption": true
  }
}
```

## Azure Application Insights Encryption
- **Service**: Application Insights `appins-cultivate-ml`
- **Data Encryption**: TLS 1.2+ in transit
- **Storage**: Encrypted at rest in Azure Monitor
- **Retention**: 90 days with encrypted storage

## File-Based Data Security
For any local development data storage:

### Environment Variables
```bash
# Encryption keys (stored in Azure Key Vault in production)
ENCRYPTION_KEY_PRIMARY=<256-bit-key>
ENCRYPTION_KEY_SECONDARY=<256-bit-key>
DATABASE_ENCRYPTION_ENABLED=true
```

### Implementation Example
```javascript
// File: src/utils/encryption.js
const crypto = require('crypto');

const ALGORITHM = 'aes-256-gcm';
const KEY_LENGTH = 32; // 256 bits
const IV_LENGTH = 16; // 128 bits

class DataEncryption {
  static encrypt(plaintext, key) {
    const iv = crypto.randomBytes(IV_LENGTH);
    const cipher = crypto.createCipher(ALGORITHM, key, iv);

    let encrypted = cipher.update(plaintext, 'utf8', 'hex');
    encrypted += cipher.final('hex');

    const authTag = cipher.getAuthTag();

    return {
      encrypted,
      iv: iv.toString('hex'),
      authTag: authTag.toString('hex')
    };
  }

  static decrypt(encryptedData, key) {
    const decipher = crypto.createDecipher(
      ALGORITHM,
      key,
      Buffer.from(encryptedData.iv, 'hex')
    );

    decipher.setAuthTag(Buffer.from(encryptedData.authTag, 'hex'));

    let decrypted = decipher.update(encryptedData.encrypted, 'hex', 'utf8');
    decrypted += decipher.final('utf8');

    return decrypted;
  }
}
```

## Azure Key Vault Integration (Production)
For production deployments, encryption keys should be managed through Azure Key Vault:

### Key Vault Configuration
- **Service**: Azure Key Vault
- **Keys**: Managed HSM or software-protected
- **Access Policy**: Service Principal with limited permissions
- **Rotation**: Automatic key rotation every 90 days

### Production Environment Variables
```bash
AZURE_KEY_VAULT_URL=https://cultivate-kv.vault.azure.net/
AZURE_CLIENT_ID=<service-principal-id>
AZURE_CLIENT_SECRET=<service-principal-secret>
AZURE_TENANT_ID=<tenant-id>
```

## Compliance & Standards
- **GDPR**: Data encryption supports GDPR compliance
- **SOC 2 Type II**: Azure storage meets SOC 2 requirements
- **ISO 27001**: Encryption follows ISO 27001 standards
- **FIPS 140-2**: Azure uses FIPS 140-2 Level 2 validated HSMs

## Security Monitoring
- **Azure Security Center**: Monitors encryption compliance
- **Application Insights**: Logs encryption operations
- **Key Vault Auditing**: Tracks key access and usage

## Verification Commands
```bash
# Check Azure Storage encryption status
az storage account show \
  --name stcultivateml \
  --resource-group cultivate-ml-rg \
  --query encryption

# Verify Key Vault access
az keyvault secret list \
  --vault-name cultivate-kv \
  --query "[].{Name:name, Enabled:attributes.enabled}"
```

## Implementation Status
- ✅ Azure Storage encryption at rest (default enabled)
- ✅ TLS 1.2+ encryption in transit
- ✅ Application Insights data encryption
- ✅ Security headers configured
- ⚠️ Key Vault integration (ready for production)
- ⚠️ Custom encryption utilities (available for sensitive data)

## Next Steps for Production
1. Set up Azure Key Vault
2. Configure service principal authentication
3. Implement automatic key rotation
4. Set up compliance monitoring dashboards