# GitHub Repository Setup Script
# Run this after creating repository on GitHub.com

# 1. Initialize Git repository
Write-Host "=== Initializing Git Repository ===" -ForegroundColor Cyan
git init

# 2. Add all files
Write-Host "`n=== Adding Files ===" -ForegroundColor Cyan
git add .

# 3. Check status
Write-Host "`n=== Git Status ===" -ForegroundColor Yellow
git status

# 4. Initial commit
Write-Host "`n=== Creating Initial Commit ===" -ForegroundColor Cyan
git commit -m "🎉 Initial release: Bridge Damage Assessment System v0.1

- Implemented 3 vision modes (llama-cpp/HuggingFace/Ollama)
- Complete pipeline: preprocessing → vision → structuring → scoring
- Validated with 10-image batch (100% success rate, 51.6s/image)
- Added comprehensive documentation (README.md, CHANGELOG.md)
- Resolved Windows encoding issues
- GPU-optimized inference with GGUF quantization

Tested on: NVIDIA RTX 4060 Ti (16GB VRAM)
Dataset: 254 rebar exposure images"

# 5. Rename branch to main
Write-Host "`n=== Setting Main Branch ===" -ForegroundColor Cyan
git branch -M main

# 6. Add remote (UPDATE THIS URL!)
Write-Host "`n=== Adding Remote ===" -ForegroundColor Cyan
Write-Host "IMPORTANT: Update the URL below with your actual repository URL!" -ForegroundColor Red
Write-Host "Example: git remote add origin https://github.com/your-username/damage_text_score.git" -ForegroundColor Yellow
Write-Host ""
$repoUrl = Read-Host "Enter your GitHub repository URL"

if ($repoUrl) {
    git remote add origin $repoUrl
    Write-Host "Remote added: $repoUrl" -ForegroundColor Green
} else {
    Write-Host "Skipped adding remote. You can add it later with:" -ForegroundColor Yellow
    Write-Host "git remote add origin <your-repo-url>" -ForegroundColor Yellow
}

# 7. Push to GitHub
Write-Host "`n=== Ready to Push ===" -ForegroundColor Cyan
Write-Host "When ready, run the following command to push to GitHub:" -ForegroundColor Yellow
Write-Host "git push -u origin main" -ForegroundColor Green

Write-Host "`n=== Git Setup Complete ===" -ForegroundColor Green
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Create repository 'damage_text_score' on GitHub.com" -ForegroundColor White
Write-Host "2. Add remote if not done: git remote add origin <url>" -ForegroundColor White
Write-Host "3. Push: git push -u origin main" -ForegroundColor White
Write-Host "4. Add GitHub Description from GITHUB_SETUP.md" -ForegroundColor White
Write-Host "5. Add Topics/Tags from GITHUB_SETUP.md" -ForegroundColor White
Write-Host "6. Create release v0.1.0 with notes from GITHUB_SETUP.md" -ForegroundColor White
