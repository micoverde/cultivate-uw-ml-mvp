# 🌿 Git Workflow Guide - Cultivate Learning ML MVP

## 📋 Branch Strategy Overview
```
feature/fix branches → dev → main → Azure SWA (Production)
```

## 🏗️ Branch Structure

### **main** (Production)
- **Protected**: ✅ Requires PR reviews + Azure CI/CD status checks
- **Purpose**: Production-ready code that deploys to Azure Static Web Apps
- **Merge Policy**: Only from `dev` branch after thorough testing
- **Auto-Deploy**: Push to main triggers Azure SWA deployment

### **dev** (Integration)
- **Protected**: ✅ Requires PR reviews (lighter requirements)
- **Purpose**: Integration testing and team collaboration
- **Merge Policy**: Feature/fix branches merge here first
- **Testing**: All features integrate here before production

### **feature/fix branches** (Development)
- **Protected**: ❌ No protection (developer freedom)
- **Purpose**: Individual feature development and bug fixes
- **Lifecycle**: Created from dev, merged back to dev, then deleted

## 🎯 Branch Naming Conventions

### Feature Branches
```bash
feature/issue-XX-description
```
**Examples:**
- `feature/issue-72-branch-strategy`
- `feature/issue-46-transcript-input`
- `feature/issue-82-backend-deployment`

### Bug Fix Branches
```bash
fix/issue-XX-description
```
**Examples:**
- `fix/issue-105-ml-model-crash`
- `fix/issue-78-responsive-layout`

### Hotfix Branches (Emergency Production Fixes)
```bash
hotfix/critical-description
```
**Examples:**
- `hotfix/security-vulnerability`
- `hotfix/deployment-failure`

## 🔄 Development Workflow

### 1. Starting New Work
```bash
# 1. Switch to dev and pull latest
git checkout dev
git pull origin dev

# 2. Create feature branch
git checkout -b feature/issue-XX-description

# 3. Make your changes
# ... code, test, commit ...

# 4. Push feature branch
git push -u origin feature/issue-XX-description
```

### 2. Creating Pull Requests
```bash
# 1. Push your latest changes
git push origin feature/issue-XX-description

# 2. Create PR to dev branch
gh pr create --base dev --title "Feature: Description" --body "Closes #XX"

# 3. Request review from team members
gh pr review --approve  # (after testing)
```

### 3. Merging to Production
```bash
# 1. Create PR from dev to main (periodically)
gh pr create --base main --head dev --title "Deploy to Production"

# 2. Final review and approval
# 3. Merge triggers Azure SWA deployment
```

## 👥 Team Collaboration

### Code Review Guidelines
- **All PRs require 1 review** (configured in branch protection)
- **Self-approval allowed for dev branch** (speed development)
- **Mandatory external review for main branch** (production safety)
- **Use GitHub's suggestion feature** for small fixes

### Merge Strategies
- **Squash and merge**: Preferred for feature branches to dev
- **Merge commit**: Preferred for dev to main (preserves history)
- **Rebase**: Never use for shared branches

## 🚀 Common Commands Reference

### Branch Management
```bash
# List all branches
git branch -a

# Switch branches
git checkout branch-name

# Create and switch to new branch
git checkout -b new-branch-name

# Delete local branch
git branch -d branch-name

# Delete remote branch
git push origin --delete branch-name
```

### Staying Up-to-Date
```bash
# Update dev branch
git checkout dev
git pull origin dev

# Rebase feature branch on latest dev
git checkout feature/issue-XX-description
git rebase dev

# Force push after rebase (only for feature branches!)
git push --force-with-lease origin feature/issue-XX-description
```

### Emergency Procedures
```bash
# Create hotfix from main
git checkout main
git pull origin main
git checkout -b hotfix/critical-issue
# Make fix, test, commit
git push -u origin hotfix/critical-issue
# Create PR directly to main
gh pr create --base main --title "HOTFIX: Critical Issue"
```

## 🔒 Branch Protection & Security Architecture

### main Branch Protection
- ✅ Require pull request reviews (1 required)
- ✅ Require code owner reviews (@micoverde approval)
- ✅ Dismiss stale PR reviews when new commits pushed
- ✅ Require status checks: "Fast CI" (immediate feedback)
- ✅ Require branches to be up-to-date before merging
- ❌ Include administrators in restrictions (for emergency access)

### dev Branch Protection
- ✅ Require pull request reviews (1 required)
- ✅ Require code owner reviews (@micoverde approval)
- ✅ Dismiss stale PR reviews when new commits pushed
- ✅ Required status checks: "Fast CI" (basic validation)
- ❌ Administrator restrictions (team flexibility)

## 🛡️ Asynchronous Security Monitoring

### Security Architecture Philosophy
**Fast Deploy + Async Security + Auto Rollback**

1. **Fast CI** (~2-3 minutes): Basic validation, syntax checks, build verification
2. **Azure Deployment** (~5 minutes): Immediate deployment to staging/production
3. **Security Monitoring** (~10-15 minutes): Comprehensive CodeQL analysis
4. **Auto-Rollback** (if needed): Emergency rollback if critical issues detected

### Security Workflows

#### Fast CI (Blocking - Required for PRs)
- ✅ Python syntax validation
- ✅ Frontend build verification
- ✅ Basic security checks (secrets, debug code)
- ⏱️ **Duration**: 2-3 minutes
- 🚦 **Blocks**: PR merging

#### Security Monitoring (Non-blocking - Async)
- 🔍 **CodeQL Analysis**: JavaScript + Python security scanning
- 🕒 **Schedule**: Every 12 hours + on every push
- 📊 **Severity**: Critical/High alerts trigger emergency response
- ⏱️ **Duration**: 10-15 minutes
- 🚦 **Blocks**: Nothing (runs asynchronously)

#### Emergency Response (Automatic)
- 🚨 **Trigger**: Critical/High severity security issues detected
- 📝 **Alert**: Creates GitHub issue with security details
- 🔄 **Rollback**: Auto-creates emergency rollback PR for main branch
- 📢 **Notification**: Alerts via GitHub + optional Slack/Teams
- ⚡ **Speed**: Immediate response upon detection

### Security Alert Response Procedure

#### Automatic Actions (No human intervention)
1. **Detection**: CodeQL finds critical security issue
2. **Alert**: GitHub issue created with details
3. **Rollback**: Emergency PR created from safe commit
4. **Notification**: Stakeholders notified

#### Manual Actions Required
1. **Review**: Check security alerts and assess impact
2. **Merge**: Approve emergency rollback PR if needed
3. **Investigate**: Root cause analysis of security issue
4. **Patch**: Fix vulnerabilities before re-deployment

## 🎯 SPRINT 2b Specific Guidelines

### Issue Integration
- **Always reference issue numbers** in branch names and PR titles
- **Use "Closes #XX"** in PR descriptions to auto-close issues
- **Link PRs to project board** items for tracking

### Story Point Tracking
- **Include story points** in PR descriptions when applicable
- **Update project board status** as PRs progress
- **Mark issues complete** only after merge to dev

### Team Assignment
- **SDE-1 (Harvard CS)**: Frontend features
- **SDE-2 (MIT CS/EE)**: Backend/API features
- **SDE-3 (MIT CS/EE)**: DevOps/Infrastructure features
- **SDE-4 (MIT AI PhD)**: ML/Research features
- **SDE-5 (MIT CS/EE)**: QA/Testing features

## ⚡ Quick Start for New Team Members

```bash
# 1. Clone repository
git clone https://github.com/micoverde/cultivate-uw-ml-mvp.git
cd cultivate-uw-ml-mvp

# 2. Set up development environment (see STORY 0.2)
npm install  # or pip install -r requirements.txt

# 3. Start working on assigned issue
git checkout dev
git pull origin dev
git checkout -b feature/issue-XX-your-assigned-task

# 4. Make changes, commit, push, create PR
git add .
git commit -m "Implement feature XX: Description"
git push -u origin feature/issue-XX-your-assigned-task
gh pr create --base dev
```

## 🆘 Troubleshooting

### "Can't push to protected branch"
- ✅ **Solution**: Create PR instead of direct push
- ❌ **Don't**: Try to force push to protected branches

### "PR checks failing"
- ✅ **Check**: Azure CI/CD logs for build/deployment errors
- ✅ **Fix**: Address issues in your feature branch
- ✅ **Update**: Push fixes and checks will re-run

### "Merge conflicts"
- ✅ **Update**: Rebase your branch on latest dev/main
- ✅ **Resolve**: Fix conflicts in your local editor
- ✅ **Test**: Ensure everything still works after resolution

---

**Last Updated**: September 2025
**Sprint**: SPRINT 2b
**Team**: 5 SDEs (Harvard CS, MIT CS/EE, MIT AI PhD)