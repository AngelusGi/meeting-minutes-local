#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

# Venv valido = esiste il suo interprete; uno parziale viene ricreato.
if [ -d .venv ] && [ ! -x .venv/bin/python ]; then
  echo "Venv incompleto o corrotto: lo ricreo..."
  rm -rf .venv
fi

if [ ! -x .venv/bin/python ]; then
  echo "Creo l'ambiente virtuale..."
  python3 -m venv .venv || { echo "ERRORE: creazione venv fallita (serve python3-venv?)"; exit 1; }
fi

source .venv/bin/activate || { echo "ERRORE: attivazione venv fallita"; exit 1; }

echo "Verifica delle dipendenze in corso..."
if ! python -c "import fastapi, uvicorn, multipart, faster_whisper, docx, imageio_ffmpeg, requests" >/dev/null 2>&1; then
  echo "Installazione in corso..."
  python -m pip install -r requirements.txt || { echo "ERRORE: installazione dipendenze fallita"; exit 1; }
fi

(
  sleep 2
  if command -v xdg-open >/dev/null 2>&1; then
    xdg-open http://127.0.0.1:8756 >/dev/null 2>&1
  elif command -v open >/dev/null 2>&1; then
    open http://127.0.0.1:8756 >/dev/null 2>&1
  else
    echo "Apri il browser su http://127.0.0.1:8756"
  fi
) &
python app.py
