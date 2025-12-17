#!/bin/bash
# Setup git hooks for vgrep development

set -e

echo "Setting up git hooks..."

# Configure git to use our hooks directory
git config core.hooksPath .github/hooks

# Make hooks executable
chmod +x .github/hooks/pre-commit
chmod +x .github/hooks/pre-push

echo "Git hooks configured successfully!"
echo "Hooks will run automatically before commits and pushes."
