# Deployment and Release Process for intura-ai

This document outlines the deployment and release process for the intura-ai package.

## Setup Required

Before you can use the automated deployment tools, you'll need to set up:

1. **PyPI API Token**: 
   - Create an account on [PyPI](https://pypi.org/)
   - Generate an API token in your account settings
   - Add the token as a secret in your GitHub repository:
     - Name: `PYPI_API_TOKEN`
     - Value: Your PyPI token

2. **Development Dependencies**:
   Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

## Release Options

You have three options for releasing new versions:

### Option 1: Using the Makefile

The Makefile provides simple commands for versioning and deployment:

```bash
# Bump version and release (choose one)
make release-patch  # For bug fixes (0.0.X)
make release-minor  # For new features (0.X.0)
make release-major  # For breaking changes (X.0.0)

# Individual steps
make bump-patch     # Just bump version
make build          # Just build package
make deploy         # Just deploy to PyPI
```

### Option 2: Using the release.py Script

The Python script provides more flexibility:

```bash
# Basic usage (defaults to patch version bump)
python release.py

# Specify version bump type
python release.py --major
python release.py --minor
python release.py --patch

# Set specific version
python release.py --version 1.2.3

# Customize the process
python release.py --no-build --no-deploy --no-tag
python release.py --dry-run  # Test without making changes
```

### Option 3: Using GitHub Actions (CI/CD)

The GitHub Actions workflow automates testing, building, and deployment:

1. Create and push a tag:
   ```bash
   # Update version in intura_ai/__version__.py first
   git add intura_ai/__version__.py
   git commit -m "Bump version to X.Y.Z"
   git tag -a vX.Y.Z -m "Version X.Y.Z"
   git push && git push --tags
   ```

2. The GitHub Actions workflow will:
   - Run tests on multiple Python versions
   - Build the package
   - Check version consistency
   - Deploy to PyPI
   - Create a GitHub release

## CHANGELOG Management

For the GitHub release notes, add entries to `CHANGELOG.md` using this format:

```markdown
# Changelog

## [0.1.0] - 2023-04-02
### Added
- New feature 1
- New feature 2

### Fixed
- Bug fix 1
- Bug fix 2
```

## Manual Release (if needed)

If you need to manually release:

```bash
# Update version in intura_ai/__version__.py
python -m build
python -m twine upload dist/*
```

## Release Checklist

Before releasing a new version, ensure:

1. All tests pass
2. Documentation is updated
3. CHANGELOG.md is updated (if you're using it)
4. Version number follows semantic versioning
5. Release notes or commit messages clearly describe changes