@echo off
cd /d "%~dp0"
echo === Minute Agent ===

where python >nul 2>nul
if errorlevel 1 (
    echo ERRORE: Python non trovato. Installalo da https://python.org e riprova.
    pause & exit /b 1
)

rem Venv valido = esiste il suo interprete, non solo la cartella.
rem Un venv parziale (creazione interrotta) viene ricreato da zero.
if exist ".venv" if not exist ".venv\Scripts\python.exe" (
    echo Venv incompleto o corrotto: lo ricreo...
    rmdir /s /q ".venv"
)

if not exist ".venv\Scripts\python.exe" (
    echo Creo l'ambiente virtuale...
    python -m venv .venv
    if errorlevel 1 (
        echo ERRORE: creazione del venv fallita. Verifica l'installazione di Python.
        pause & exit /b 1
    )
)

call ".venv\Scripts\activate.bat"
if errorlevel 1 (
    echo ERRORE: attivazione del venv fallita.
    pause & exit /b 1
)

echo Verifica delle dipendenze in corso...
python -c "import fastapi, uvicorn, multipart, faster_whisper, docx, imageio_ffmpeg, requests" >nul 2>nul
if errorlevel 1 (
    echo Installazione in corso...
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo ERRORE: installazione dipendenze fallita. Controlla la connessione e rilancia.
        pause & exit /b 1
    )
)

start "" http://127.0.0.1:8756
echo Avvio del server su http://127.0.0.1:8756 ... ^(CTRL+C per uscire^)
python app.py
pause
