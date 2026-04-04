# MovieMind Startup Script

# 1. Check for .env file and AI Key
$envPath = "backend\.env"
if (Test-Path $envPath) {
    $content = Get-Content $envPath
    if ($content -match "your_api_key_here" -and $content -match "your_openrouter_key_here") {
        Write-Host "--------------------------------------------------------" -ForegroundColor Yellow
        Write-Host "ACTION REQUIRED: No AI Key Found!" -ForegroundColor Yellow
        Write-Host "The system is currently using Templates mode."
        Write-Host "To enable the full AI, please add your key to: $envPath"
        Write-Host "Get a free key here: https://aistudio.google.com/apikey"
        Write-Host "--------------------------------------------------------"
    } else {
        Write-Host "AI Key Detected! Initializing Brain..." -ForegroundColor Green
    }
}

# 1.5. Check for Ollama
$ollamaUrl = "http://localhost:11434/api/tags"
try {
    $resp = Invoke-RestMethod -Uri $ollamaUrl -Method Get -ErrorAction Stop
    $models = $resp.models | ForEach-Object { $_.name }
    
    $confModel = "mistral"
    if (Test-Path $envPath) {
        $envLines = Get-Content $envPath
        foreach ($line in $envLines) {
            if ($line -like "OLLAMA_MODEL=*") {
                $rawModel = $line.Split("=")[1].Trim()
                $confModel = $rawModel.Replace('"', '').Replace("'", "")
                break
            }
        }
    }

    if ($models -contains $confModel -or $models -contains "$confModel:latest") {
        Write-Host "Ollama detected with model '$confModel'. Local AI ready." -ForegroundColor Green
    } else {
        Write-Host "Ollama model '$confModel' not found. Run: ollama pull $confModel" -ForegroundColor Yellow
    }
} catch {
    Write-Host "Ollama not detected. Local AI disabled." -ForegroundColor Gray
}

# 2. Check for Venv
if (-not (Test-Path "backend\venv")) {
    Write-Host "Virtual environment not found in backend\venv." -ForegroundColor Red
    exit
}

# 3. Start the Server
Write-Host "Starting MovieMind Backend..." -ForegroundColor Cyan
Set-Location "backend"
.\venv\Scripts\python.exe app.py
