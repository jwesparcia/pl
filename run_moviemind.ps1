# MovieMind Startup Script

# 1. Check for .env file and API Key
$envPath = "backend\.env"
if (Test-Path $envPath) {
    $content = Get-Content $envPath
    if ($content -match "your_api_key_here" -and $content -match "your_openrouter_key_here") {
        Write-Host "--------------------------------------------------------" -ForegroundColor Yellow
        Write-Host "⚠️  ACTION REQUIRED: No AI Key Found!" -ForegroundColor Yellow
        Write-Host "The system is currently using 'Safe Mode' (Templates)."
        Write-Host "To enable the full AI Brain, please add your key to: $envPath"
        Write-Host "Get a free key here: https://aistudio.google.com/apikey"
        Write-Host "--------------------------------------------------------"
    } else {
        Write-Host "✅ AI Key Detected! Initializing Brain..." -ForegroundColor Green
    }
}

# 2. Check for Venv
if (-not (Test-Path "backend\venv")) {
    Write-Host "❌ Virtual environment not found in backend\venv." -ForegroundColor Red
    exit
}

# 3. Start the Server
Write-Host "Starting MovieMind Backend..." -ForegroundColor Cyan
cd backend
.\venv\Scripts\python.exe app.py
