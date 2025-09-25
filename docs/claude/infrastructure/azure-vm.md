# Azure VM Testing Infrastructure

**Purpose**: Complete documentation for Azure VM-based testing and development infrastructure
**Last Updated**: 2025-09-24
**Related**: [CI/CD Deployment Architecture](../cicd/deployment-architecture.md), [Distributed AI Development](../ai/distributed-development.md)

---

## 🖥️ Primary Development VM

**Connection Details**:
```bash
ssh azureuser@20.185.221.53
```

**Architecture**: Standard Azure Linux VM optimized for development workloads
**Purpose**: Long-duration fuzz testing, production validation, parallel branch testing
**Status**: Active 24/7 for continuous testing operations

## 📁 VM Directory Structure

### **Main Production Repository** (`~/src/scratchjr-web-unified`)
- **Branch**: `main` (production)
- **Purpose**: Production stability testing and main branch validation
- **Port**: 9995 (for production fuzz testing)
- **Git Remote**: Origin main branch

### **Development Repository** (`~/src/scratchjr-dev`)
- **Branch**: `main-r`
- **Purpose**: Active development branch testing and feature validation
- **Port**: 9996 (for main-r branch testing)
- **Git Remote**: Origin main-r branch

## 🚀 Fuzz Testing Operations

### **Quick Test Commands**
```bash
# 30-second validation test
ssh azureuser@20.185.221.53 "cd ~/src/scratchjr-web-unified && node tests/fuzz/fuzz-runner.js --duration 0.5"

# 1-minute test with server
ssh azureuser@20.185.221.53 "cd ~/src/scratchjr-web-unified && PORT=9998 npm run serve > test-server.log 2>&1 & sleep 5 && node tests/fuzz/fuzz-runner.js --duration 1"
```

### **Long-Duration Production Tests**
```bash
# 2-hour production test with nohup
ssh azureuser@20.185.221.53 "cd ~/src/scratchjr-web-unified && nohup bash -c 'PORT=9999 npm run serve > server.log 2>&1 & sleep 10 && node tests/fuzz/fuzz-runner.js --duration 120' > fuzz-session.log 2>&1 &"

# 5-hour parallel testing (main + main-r branches)
ssh azureuser@20.185.221.53 "cd ~/src/scratchjr-web-unified && nohup node tests/fuzz/fuzz-runner.js --duration 300 > prod-main-fuzz-5hr.log 2>&1 &"
ssh azureuser@20.185.221.53 "cd ~/src/scratchjr-dev && nohup node tests/fuzz/fuzz-runner.js --duration 300 > prod-main-r-fuzz-5hr.log 2>&1 &"
```

## 🔍 Monitoring and Process Management

### **Check Running Processes**
```bash
# Check all active fuzz tests
ssh azureuser@20.185.221.53 "ps aux | grep -E 'node.*fuzz' | grep -v grep"

# Check server processes
ssh azureuser@20.185.221.53 "ps aux | grep -E 'npm.*serve' | grep -v grep"

# Comprehensive process check
ssh azureuser@20.185.221.53 "ps aux | grep -E 'node|npm' | grep -v grep"
```

### **Log File Monitoring**
```bash
# Real-time log following
ssh azureuser@20.185.221.53 "tail -f ~/src/scratchjr-web-unified/prod-main-fuzz-5hr.log"

# Check recent log entries
ssh azureuser@20.185.221.53 "tail -10 ~/src/scratchjr-web-unified/prod-main-fuzz-5hr.log"

# Check main-r branch progress
ssh azureuser@20.185.221.53 "tail -5 ~/src/scratchjr-dev/prod-main-r-fuzz-5hr.log"
```

### **Process Control**
```bash
# Gracefully stop fuzz processes
ssh azureuser@20.185.221.53 "pkill -f 'fuzz-runner.js'"

# Stop server processes
ssh azureuser@20.185.221.53 "pkill -f 'npm run serve'"

# Verify all stopped
ssh azureuser@20.185.221.53 "ps aux | grep -E 'node|npm' | grep -v grep"
```

## 📊 Log File Locations

### **Main Branch Testing**
```bash
~/src/scratchjr-web-unified/prod-main-fuzz-5hr.log     # 5-hour production test
~/src/scratchjr-web-unified/server-main.log            # Server output
~/src/scratchjr-web-unified/fuzz-2hr-production.log    # 2-hour production test
```

### **Development Branch Testing**
```bash
~/src/scratchjr-dev/prod-main-r-fuzz-5hr.log          # 5-hour main-r test
~/src/scratchjr-dev/server-main-r.log                 # main-r server output
~/src/scratchjr-dev/fuzz-main-r-1hr.log               # 1-hour main-r test
```

### **Quick Test Logs**
```bash
~/src/scratchjr-web-unified/test-server.log           # Quick test server
~/src/scratchjr-web-unified/remote-test.log           # Remote execution test
```

## 🔧 Repository Management on VM

### **Updating Code**
```bash
# Update main repository
ssh azureuser@20.185.221.53 "cd ~/src/scratchjr-web-unified && git fetch origin && git pull origin main"

# Update main-r repository
ssh azureuser@20.185.221.53 "cd ~/src/scratchjr-dev && git fetch origin && git pull origin main-r"

# Check current branch status
ssh azureuser@20.185.221.53 "cd ~/src/scratchjr-web-unified && git status"
```

### **Branch Management**
```bash
# Switch to specific branch
ssh azureuser@20.185.221.53 "cd ~/src/scratchjr-web-unified && git checkout [branch-name]"

# See all available branches
ssh azureuser@20.185.221.53 "cd ~/src/scratchjr-web-unified && git branch -a"

# Check recent commits
ssh azureuser@20.185.221.53 "cd ~/src/scratchjr-web-unified && git log --oneline -5"
```

## 🚨 SSH Connection Best Practices

### **Connection Handling**
- **Expected Behavior**: SSH timeouts are normal with long-running background processes
- **Reconnection**: Simply run `ssh azureuser@20.185.221.53` again
- **Fast Timeout**: Use `-o ConnectTimeout=30` for quicker timeout detection
- **Background Processes**: Always use `nohup` for long-duration tests

### **nohup Usage Patterns**
```bash
# Single command with nohup
nohup command > logfile.log 2>&1 &

# Complex command sequence with nohup
nohup bash -c 'PORT=9999 npm run serve > server.log 2>&1 & sleep 10 && node tests/fuzz/fuzz-runner.js --duration 120' > combined.log 2>&1 &

# Check nohup process status
jobs
ps aux | grep [process-name]
```

## 💻 Environment Configuration

### **Node.js and Dependencies**
```bash
# Check versions
ssh azureuser@20.185.221.53 "node --version && npm --version"

# Install dependencies if needed
ssh azureuser@20.185.221.53 "cd ~/src/scratchjr-web-unified && npm install"

# Build project
ssh azureuser@20.185.221.53 "cd ~/src/scratchjr-web-unified && npm run build"
```

### **GitHub CLI Integration**
```bash
# Check GitHub authentication
ssh azureuser@20.185.221.53 "gh auth status"

# List recent issues
ssh azureuser@20.185.221.53 "gh issue list --limit 5"

# View specific issue
ssh azureuser@20.185.221.53 "gh issue view 218"
```

## 🎯 Common VM Operations

### **Quick Status Check**
```bash
ssh azureuser@20.185.221.53 "echo 'VM Status Check:' && date && echo 'Active Processes:' && ps aux | grep -E 'node.*fuzz|npm.*serve' | grep -v grep && echo 'Recent Logs:' && tail -3 ~/src/scratchjr-web-unified/prod-main-fuzz-5hr.log 2>/dev/null || echo 'No main log found'"
```

### **Emergency Process Cleanup**
```bash
# Kill all Node.js processes (CAUTION!)
ssh azureuser@20.185.221.53 "pkill node"

# Kill all npm processes
ssh azureuser@20.185.221.53 "pkill npm"

# Verify cleanup
ssh azureuser@20.185.221.53 "ps aux | grep -E 'node|npm' | grep -v grep"
```

### **Port Management**
```bash
# Check what's using a specific port
ssh azureuser@20.185.221.53 "lsof -i :9999"

# Kill process using specific port
ssh azureuser@20.185.221.53 "lsof -ti :9999 | xargs kill -9"
```

## 🏗️ Infrastructure Use Cases

### **Long-Duration Testing**
- **5+ hour continuous fuzz tests** for stability validation
- **Parallel main/main-r branch testing** for regression detection
- **Production environment simulation** under realistic load

### **Feature Validation**
- **Enhanced bug reporting testing** with new fingerprinting system
- **Deduplication system validation** across multiple test runs
- **Branch comparison analysis** for feature impact assessment

### **Emergency Response**
- **Rapid bug reproduction** in isolated environment
- **Quick fix validation** before production deployment
- **Rollback testing** for critical issues

## ⚡ VM Quick Reference Card

| Operation | Command |
|-----------|---------|
| **Connect** | `ssh azureuser@20.185.221.53` |
| **Quick Test** | `cd ~/src/scratchjr-web-unified && node tests/fuzz/fuzz-runner.js --duration 0.5` |
| **Long Test** | `nohup node tests/fuzz/fuzz-runner.js --duration 120 > test.log 2>&1 &` |
| **Check Processes** | `ps aux \| grep -E 'node.*fuzz' \| grep -v grep` |
| **View Logs** | `tail -f prod-main-fuzz-5hr.log` |
| **Stop Tests** | `pkill -f 'fuzz-runner.js'` |
| **Update Code** | `git fetch origin && git pull origin main` |
| **Check Status** | `git status && ps aux \| grep node` |

This VM infrastructure enables continuous validation of ScratchJr improvements with real-world testing scenarios that would be impractical on local development machines.

---

**Navigation**: [← Main Memory](../../CLAUDE.md) | [CI/CD Deployment →](../cicd/deployment-architecture.md)