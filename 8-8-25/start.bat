@echo off
set PROJ=C:\Users\tossy\OneDrive\CIV\TESTING\8-8-25
cd /d "%PROJ%"

if not exist .venv (
  py -3 -m venv .venv
)

call .venv\Scripts\activate.bat
pip install -r requirements.txt
streamlit run ui\app.py
