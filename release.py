#!/usr/bin/env python3
"""
Automated release script for intura-ai.

This script automates the process of versioning, building, and releasing
the intura-ai package to PyPI.

Usage:
    python release.py [--major | --minor | --patch | --version VERSION]
                     [--no-build] [--no-deploy] [--no-tag] [--dry-run]

Options:
    --major             Bump major version (X.0.0)
    --minor             Bump minor version (0.X.0)
    --patch             Bump patch version (0.0.X) [default]
    --version VERSION   Set specific version
    --no-build          Skip building package
    --no-deploy         Skip deploying to PyPI
    --no-tag            Skip creating Git tag
    --dry-run           Perform a dry run (no actual changes)
"""

import os
import re
import sys
import argparse
import subprocess
from typing import Tuple, Optional, List

# Constants
VERSION_FILE = "intura_ai/__version__.py"
VERSION_PATTERN = r'__version__\s*=\s*"(.+?)"'


def run_command(command: List[str], dry_run: bool = False) -> int:
    """Run a shell command."""
    print(f"Running: {' '.join(command)}")
    if dry_run:
        return 0
    return subprocess.call(command)


def get_current_version() -> str:
    """Get the current version from __version__.py."""
    if not os.path.exists(VERSION_FILE):
        print(f"Error: {VERSION_FILE} not found!")
        sys.exit(1)

    with open(VERSION_FILE, "r") as f:
        content = f.read()

    match = re.search(VERSION_PATTERN, content)
    if not match:
        print(f"Error: Version pattern not found in {VERSION_FILE}!")
        sys.exit(1)

    return match.group(1)


def parse_version(version: str) -> Tuple[int, int, int]:
    """Parse version string into major, minor, patch components."""
    try:
        parts = version.split('.')
        if len(parts) >= 3:
            return int(parts[0]), int(parts[1]), int(parts[2])
        elif len(parts) == 2:
            return int(parts[0]), int(parts[1]), 0
        else:
            return int(parts[0]), 0, 0
    except (ValueError, IndexError):
        print(f"Error: Invalid version format: {version}")
        sys.exit(1)


def bump_version(current: str, major: bool = False, minor: bool = False, patch: bool = False,
                 specific: Optional[str] = None) -> str:
    """Calculate the new version based on bump type or specific version."""
    if specific:
        return specific

    major_num, minor_num, patch_num = parse_version(current)

    if major:
        return f"{major_num + 1}.0.0"
    elif minor:
        return f"{major_num}.{minor_num + 1}.0"
    elif patch:
        return f"{major_num}.{minor_num}.{patch_num + 1}"
    else:
        # Default to patch bump
        return f"{major_num}.{minor_num}.{patch_num + 1}"


def update_version_file(new_version: str, dry_run: bool = False) -> None:
    """Update the version in __version__.py."""
    if dry_run:
        print(f"Would update {VERSION_FILE} to version {new_version}")
        return

    with open(VERSION_FILE, "w") as f:
        f.write(f'__version__ = "{new_version}"\n')
    
    print(f"Updated {VERSION_FILE} to version {new_version}")


def build_package(dry_run: bool = False) -> bool:
    """Build the package."""
    print("Building package...")
    if run_command(["python", "-m", "build"], dry_run) != 0:
        print("Error: Package build failed!")
        return False
    return True


def deploy_to_pypi(dry_run: bool = False) -> bool:
    """Deploy the package to PyPI."""
    print("Deploying to PyPI...")
    if run_command(["python", "-m", "twine", "upload", "dist/*"], dry_run) != 0:
        print("Error: PyPI upload failed!")
        return False
    return True


def create_git_tag(version: str, dry_run: bool = False) -> bool:
    """Create and push a git tag for the release."""
    tag_name = f"v{version}"
    
    # Add version file
    if run_command(["git", "add", VERSION_FILE], dry_run) != 0:
        print("Error: Failed to add version file to git!")
        return False
    
    # Commit version change
    if run_command(["git", "commit", "-m", f"Bump version to {version}"], dry_run) != 0:
        print("Error: Failed to commit version change!")
        return False
    
    # Create tag
    if run_command(["git", "tag", "-a", tag_name, "-m", f"Version {version}"], dry_run) != 0:
        print("Error: Failed to create git tag!")
        return False
    
    # Push changes
    if run_command(["git", "push", "origin", "main"], dry_run) != 0:
        print("Error: Failed to push commits!")
        return False
    
    # Push tag
    if run_command(["git", "push", "origin", tag_name], dry_run) != 0:
        print("Error: Failed to push git tag!")
        return False
    
    print(f"Created and pushed git tag: {tag_name}")
    return True


def main():
    """Main function to handle the release process."""
    parser = argparse.ArgumentParser(description="Release automation for intura-ai package")
    
    version_group = parser.add_mutually_exclusive_group()
    version_group.add_argument("--major", action="store_true", help="Bump major version")
    version_group.add_argument("--minor", action="store_true", help="Bump minor version")
    version_group.add_argument("--patch", action="store_true", help="Bump patch version (default)")
    version_group.add_argument("--version", help="Set specific version")
    
    parser.add_argument("--no-build", action="store_true", help="Skip building package")
    parser.add_argument("--no-deploy", action="store_true", help="Skip deploying to PyPI")
    parser.add_argument("--no-tag", action="store_true", help="Skip creating Git tag")
    parser.add_argument("--dry-run", action="store_true", help="Perform a dry run (no actual changes)")
    
    args = parser.parse_args()
    
    # Get current version
    current_version = get_current_version()
    print(f"Current version: {current_version}")
    
    # Calculate new version
    new_version = bump_version(
        current_version,
        major=args.major,
        minor=args.minor,
        patch=args.patch or not (args.major or args.minor or args.version),
        specific=args.version
    )
    print(f"New version: {new_version}")
    
    # Update version file
    update_version_file(new_version, args.dry_run)
    
    # Build package
    if not args.no_build:
        if not build_package(args.dry_run):
            sys.exit(1)
    
    # Deploy to PyPI
    if not args.no_deploy:
        if not deploy_to_pypi(args.dry_run):
            sys.exit(1)
    
    # Create and push git tag
    if not args.no_tag:
        if not create_git_tag(new_version, args.dry_run):
            sys.exit(1)
    
    print(f"Release {new_version} completed successfully!")


if __name__ == "__main__":
    main()