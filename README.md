# Codex Curator" 
## Installation
### Reguirements
- [Python3](https://www.python.org/) (remember to **add to PATH** while installing)
- [Ollama](https://ollama.com/download)

### Installation Instructions
#### Automatic installation
If you have a bash shell run
```bash
sh setup.bash
```
#### Manual setup
**1. Init the virtual environment**
```
python3 -m venv env
```
**2. Activate the environment**:

**Windows:**

`env\Scripts\activate`

**Linux / MacOS**

`source env/bin/activate`

**3. Pull models and finalize**
```bash
ollama pull deepseek-r1:latest
ollama pull mxbai-embed-large
pip install -r requirements.txt
mkdir knowledge_base
```
## Use
If you have a bash shell just run:
```bash
sh run.bash
```

Else, first *activate the venv* as you can see in the above section and then run:

```bash
streamlit run chat_app.py
```

To add a different deepseek model (only deepseek supported atm) you need to 
* Pull the model
`ollama pull deepseek-r1:{name}`
* Run the script with the new model name 
`streamlit run chat_app.py {name}`


e.g for the 32b model 

```bash
ollama pull deepseek-r1:32b
streamlit run chat_app.py 32b
```