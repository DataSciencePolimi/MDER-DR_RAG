# MDER-DR_RAG

# Docs and Diagram

The folder `docs` is prepared to include relevant documentation and diagrams for the project.

Right now it contains the current _architecture diagram_.

## Setup

Install [conda](https://docs.conda.io/projects/conda/en/23.1.x/user-guide/install/) and [homebrew](https://brew.sh/) if needed.

_Note to self_: conda is only needed to use the same version of python as the server. Is that necessary?

create a conda environment using

```shell
conda create -n mder-dr python=3.13
```

activate it using

```shell
conda activate mder-dr
```

install the pip packages from the _requirements_ in the env

```shell
pip install -r requirements.txt
```

## Private Settings

In order to run the server, you need to create a file called private_settings.py in the same directory as settings.py. This file should contain the following variables:

```python
PRIVATE_SETINGS = {
    "LLM_LOCAL": True, # Set to True if you are using a local LLM or False if you are using a remote LLM
    "LLM_KEY": {
        "openai": "" # OpenAI API key
        "ollama": "", # ollama API key
        "anthropic": "", # Anthropic API key
        "deepseek": "", # DeepSeeker API key
    },
    "LLM_BASE_URL": "", # Base URL for the LLM local API
}
```

You can use standard urls for local deployment:

-   Ollama: `"LLM_BASE_URL": "http://localhost:11434"`
-   LM Studio: `"LLM_BASE_URL": "http://localhost:1234/v1"`

## Local LLMs

Right now, test locally with [Ollama](https://ollama.com/)

Models tried:

-   llama3.2
-   mistral

Embeddings:

-   mxbai-embed-large
-   nomic-embed-text

In order to run Ollama, launch the Ollama server in a separate terminal:

```shell
ollama run llama3.2 # or mistral
```

## UI

To run the UI

```shell
streamlit run streamlit_ui.py
```


# Citation

```
@InProceedings{10.1007/978-3-031-97207-2_4,
author="Campi, Riccardo
and Pinciroli Vago, Nicol{\`o} Oreste
and Giudici, Mathyas
and Rodriguez-Guisado, Pablo Barrachina
and Brambilla, Marco
and Fraternali, Piero",
editor="Verma, Himanshu
and Bozzon, Alessandro
and Mauri, Andrea
and Yang, Jie",
title="A Graph-Based RAG for Energy Efficiency Question Answering",
booktitle="Web Engineering",
year="2026",
publisher="Springer Nature Switzerland",
address="Cham",
pages="41--55",
abstract="In this work, we investigate the use of Large Language Models (LLMs) within a Graph-based Retrieval Augmented Generation (RAG) architecture for Energy Efficiency (EE) Question Answering. First, the system automatically extracts a Knowledge Graph (KG) from guidance and regulatory documents in the energy field. Then, the generated graph is navigated and reasoned upon to provide users with accurate answers in multiple languages. We implement a human-based validation using the RAGAs framework properties, a validation dataset composed of 101 question-answer pairs, and some domain experts. Results confirm the potential of this architecture and identify its strengths and weaknesses. Validation results show how the system correctly answers in about three out of four of the cases ({\$}{\$}75.2{\backslash}pm 2.7{\backslash}{\%}{\$}{\$}75.2{\textpm}2.7{\%}), with higher results on questions related to more general EE answers (up to {\$}{\$}81.0{\backslash}pm 4.1{\backslash}{\%}{\$}{\$}81.0{\textpm}4.1{\%}), and featuring promising multilingual abilities ({\$}{\$}4.4{\backslash}{\%}{\$}{\$}4.4{\%}accuracy loss due to translation).",
isbn="978-3-031-97207-2"
}
```
