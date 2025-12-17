# Setup git hooks for vgrep development (Windows)

Write-Host "Setting up git hooks..." -ForegroundColor Cyan

# Configure git to use our hooks directory
git config core.hooksPath .github/hooks

Write-Host "Git hooks configured successfully!" -ForegroundColor Green
Write-Host "Hooks will run automatically before commits and pushes."
