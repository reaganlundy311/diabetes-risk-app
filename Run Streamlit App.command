#!/bin/zsh
cd "$(dirname "$0")"

echo "Installing project requirements if needed..."
python3 -m pip install -r requirements.txt

echo ""
echo "Opening the Diabetes Risk Streamlit app..."
python3 -m streamlit run app/streamlit_app.py
