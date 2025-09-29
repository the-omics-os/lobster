#!/bin/bash
set -euo pipefail

# Release script for lobster package
VERSION=${1:-}

if [ -z "$VERSION" ]; then
    echo "Usage: ./scripts/release.sh <version>"
    echo "Example: ./scripts/release.sh 1.0.0"
    exit 1
fi

echo "Preparing release v$VERSION"
echo "=========================="

# Ensure we're on main branch
BRANCH=$(git branch --show-current)
if [ "$BRANCH" != "main" ]; then
    echo "Error: Releases must be created from main branch"
    echo "Current branch: $BRANCH"
    exit 1
fi

# Ensure working directory is clean
if [ -n "$(git status --porcelain)" ]; then
    echo "Error: Working directory has uncommitted changes"
    exit 1
fi

# Pull latest changes
echo "Pulling latest changes..."
git pull origin main

# Run tests if they exist
if [ -f pytest.ini ] || [ -d tests ]; then
    echo "Running tests..."
    python -m pytest tests/ || exit 1
fi

# Create and push tag
echo "Creating tag v$VERSION..."
git tag -a "v$VERSION" -m "Release version $VERSION"

echo "Pushing tag to origin..."
git push origin "v$VERSION"

echo ""
echo "Release v$VERSION created successfully!"
echo "GitHub Actions will now build and publish the package."
echo ""
echo "Monitor the release at:"
echo "https://github.com/the-omics-os/lobster/actions"
