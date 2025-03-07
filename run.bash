#!/bin/bash

if [[ "$OSTYPE" == "msys" ]]; then
    # Windows
    source env/Scripts/activate
else
    # Posix
    source env/bin/activate
fi

pip install -r requirements.txt
streamlit run chat_app.py