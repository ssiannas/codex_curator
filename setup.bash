#!/bin/bash

if [[ "$OSTYPE" == "msys" ]]; then
    # Windows
    source env/Scripts/activate
else
    # Posix
    source env/bin/activate
fi

ollama pull deepseek-r1:latest
ollama pull mxbai-embed-large
pip install -r requirements.txt
mkdir knowledge_base