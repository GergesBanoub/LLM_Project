#Mini-rag
this is a minimmal implementation of the RAG model for question and answering .

## Requirements 

- Python 3.8 or Later version


### Install Python Miniconda
1) Download and install MiniConda from ( wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh)
2) Create New virtual Environment using the following command 
```bash
$ conda create -n mini-rag python=3.8
```

3) Activate the Virtual Env 

```bash
$ conda activate mini-rag
```
## Installation 

### Install Required Packages 

```bash
$ pip install -r Requirements.txt
```
## Run the FastAPI server 
```bash
$ uvicorn main:app --reload --host 0.0.0.0 --port 5000
```

## Run the the following Code To insatll LangChain Lib 
 ```bash
$ pip install -U langchain-community
```