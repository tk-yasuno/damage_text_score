# 段階的テスト実行スクリプト
# 1枚 → 10枚 → 50枚 → 254枚 の順で実行

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "Damage Analysis System - Progressive Test" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""

# 段階1: 単一画像テスト（1枚）
Write-Host "`n[Stage 1/4] Single Image Test (1 image)" -ForegroundColor Yellow
Write-Host ("=" * 70)
python quickstart.py --mode 1

if ($LASTEXITCODE -ne 0) {
    Write-Host "`nError at Stage 1" -ForegroundColor Red
    exit 1
}

Write-Host "`nStage 1 Completed" -ForegroundColor Green
Write-Host "Press Enter to continue to Stage 2 (10 images)..."
Read-Host

# 段階2: 小規模テスト（10枚）
Write-Host "`n[Stage 2/4] Small Scale Test (10 images)" -ForegroundColor Yellow
Write-Host ("=" * 70)
python quickstart.py --mode 2

if ($LASTEXITCODE -ne 0) {
    Write-Host "`nError at Stage 2" -ForegroundColor Red
    exit 1
}

Write-Host "`nStage 2 Completed" -ForegroundColor Green
Write-Host "Press Enter to continue to Stage 3 (50 images)..."
Read-Host

# 段階3: 中規模テスト（50枚）
Write-Host "`n[Stage 3/4] Medium Scale Test (50 images)" -ForegroundColor Yellow
Write-Host ("=" * 70)
python quickstart.py --mode 3

if ($LASTEXITCODE -ne 0) {
    Write-Host "`nError at Stage 3" -ForegroundColor Red
    exit 1
}

Write-Host "`nStage 3 Completed" -ForegroundColor Green
Write-Host "Press Enter to continue to Stage 4 (254 images)..."
Read-Host

# 段段4: 全画像処理（254枚）
Write-Host "`n[Stage 4/4] Full Dataset Processing (254 images)" -ForegroundColor Yellow
Write-Host ("=" * 70)
python quickstart.py --mode 4

if ($LASTEXITCODE -ne 0) {
    Write-Host "`nError at Stage 4" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host ("=" * 70) -ForegroundColor Cyan
Write-Host "All Stages Completed!" -ForegroundColor Green
Write-Host ("=" * 70) -ForegroundColor Cyan
Write-Host "`nResult Files:"
Get-ChildItem data\outputs\*.csv | Select-Object Name, Length, LastWriteTime | Format-Table
