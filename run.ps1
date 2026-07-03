$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
Write-Host "=== Minute Agent ===" -ForegroundColor Cyan

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "ERRORE: Python non trovato. Installalo da https://python.org e riprova." -ForegroundColor Red
    Read-Host "Premi INVIO per chiudere"; exit 1
}

# Venv valido = esiste il suo interprete; uno parziale viene ricreato
if ((Test-Path ".venv") -and -not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Venv incompleto o corrotto: lo ricreo..."
    Remove-Item -Recurse -Force ".venv"
}
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Creo l'ambiente virtuale..."
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERRORE: creazione del venv fallita." -ForegroundColor Red
        Read-Host "Premi INVIO per chiudere"; exit 1
    }
}
$py = ".venv\Scripts\python.exe"

Write-Host "Verifica delle dipendenze in corso..."
& $py -c "import fastapi, uvicorn, multipart, faster_whisper, docx, imageio_ffmpeg, requests" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installazione in corso..."
    & $py -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERRORE: installazione dipendenze fallita. Controlla la connessione e rilancia." -ForegroundColor Red
        Read-Host "Premi INVIO per chiudere"; exit 1
    }
}

# Ollama: necessario per la stesura della minuta
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    $r = Read-Host "Ollama non trovato: serve per generare la minuta. Installarlo adesso con winget? [S/n]"
    if ($r -eq "" -or $r -match "^[SsYy]") {
        if (Get-Command winget -ErrorAction SilentlyContinue) {
            winget install -e --id Ollama.Ollama --accept-source-agreements --accept-package-agreements
            Write-Host "Se il comando 'ollama' non viene riconosciuto, chiudi e rilancia questo script."
        } else {
            Write-Host "winget non disponibile: scarica Ollama da https://ollama.com/download" -ForegroundColor Yellow
        }
    } else {
        Write-Host "Ok: senza Ollama otterrai trascrizione + template di minuta da compilare."
    }
}

# Modello LLM di default
if (Get-Command ollama -ErrorAction SilentlyContinue) {
    $models = ollama list 2>$null | Out-String
    if ($models -notmatch "llama3\.1:8b") {
        $r = Read-Host "Modello llama3.1:8b non presente (~4.9 GB). Scaricarlo adesso? [S/n]"
        if ($r -eq "" -or $r -match "^[SsYy]") { ollama pull llama3.1:8b }
    }
}

Start-Process "http://127.0.0.1:8756"
Write-Host "Avvio del server su http://127.0.0.1:8756 (CTRL+C per uscire)"
& $py app.py
