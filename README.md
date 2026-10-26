# 🧠 Eddie Twin — Personal Digital Twin Powered by RAG

<p align="center">
  <strong>A production-oriented Retrieval-Augmented Generation (RAG) system for building a searchable, source-grounded AI representation of personal knowledge.</strong>
</p>

<p align="center">
  <a href="https://eddie-twin-6qwt6zd6rnybpkndhfjaxo.streamlit.app/">
    <img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Open Eddie Twin" />
  </a>
  <a href="https://github.com/EddiesEdit/eddie-twin">
    <img src="https://img.shields.io/badge/GitHub-Repository-181717?logo=github" alt="GitHub repository" />
  </a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+" />
  <img src="https://img.shields.io/badge/Streamlit-Deployed-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit" />
  <img src="https://img.shields.io/badge/Vector%20Database-Qdrant-DC244C" alt="Qdrant" />
  <img src="https://img.shields.io/badge/Tests-97%20passing-success" alt="97 passing tests" />
</p>

---

## 🚀 Live Demo

**[Try Eddie Twin](https://eddie-twin-6qwt6zd6rnybpkndhfjaxo.streamlit.app/)**

Eddie Twin is a personal AI knowledge assistant designed to answer questions using information retrieved from uploaded documents. Rather than relying exclusively on a language model's pretrained knowledge, the application retrieves relevant document content and uses it as context for generating responses.

The project explores how to engineer a maintainable RAG application with modular components, vector search, document metadata, input validation, security defenses, automated testing, and a deployed Streamlit interface.

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [How It Works](#-how-it-works)
- [Architecture](#️-architecture)
- [Technology Stack](#️-technology-stack)
- [Getting Started](#-getting-started)
- [Environment Configuration](#-environment-configuration)
- [Running the Application](#-running-the-application)
- [Running Tests](#-running-tests)
- [Security and Responsible Use](#-security-and-responsible-use)
- [Deployment](#-deployment)
- [Current Limitations](#-current-limitations)
- [Roadmap](#-roadmap)
- [Author](#-author)

## 🎯 Project Overview

Eddie Twin began as an experiment in building a personal digital twin from structured and unstructured knowledge, starting with my CV and expanding toward a reusable personal knowledge base.

The goal is to move beyond a basic chatbot and build an AI system with explicit ingestion, retrieval, generation, metadata management, validation, and testing boundaries.

The system follows a modular, object-oriented design so its components can be developed and tested independently.

### Why this project?

Traditional chatbots may generate answers that sound plausible without being grounded in a user's actual documents. A RAG system helps address this by retrieving relevant information at query time and supplying it to the language model as context.

Eddie Twin brings these concepts together in a practical application that can:

- Ingest PDF documents.
- Extract and split document text into manageable chunks.
- Generate embeddings for semantic search.
- Store vectors and metadata in Qdrant.
- Retrieve relevant document content for a question.
- Generate responses using an OpenAI language model.
- Display source information to support answer traceability.

## ✨ Key Features

### 📄 Document ingestion

Upload PDF documents through the Streamlit sidebar and process them into searchable knowledge.

The ingestion pipeline extracts document text, creates chunks, generates embeddings, and stores vectors with associated metadata.

### 🔎 Semantic retrieval

Eddie Twin uses OpenAI embeddings and Qdrant vector search to identify document chunks relevant to a user's question.

This enables retrieval based on semantic similarity rather than relying only on exact keyword matches.

### 🤖 Context-aware generation

Retrieved document content is passed to an OpenAI language model to generate a response informed by the available evidence.

### 📚 Source traceability

Responses can include source information, such as document filenames and page numbers, helping users inspect where retrieved information originated.

### 🛡️ Security-oriented design

The project includes input validation, prompt-injection defenses, output checks, and audit practices designed to reduce exposure to unsafe inputs and outputs.

These controls are defense-in-depth measures; they do not guarantee that every attack or hallucination will be prevented.

### 🧪 Automated testing

The project has a test suite covering important components and behaviors. The current verified baseline is **97 passing tests**.

### ☁️ Deployed web interface

The Streamlit interface provides document upload, ingestion controls, a conversational question-and-answer experience, and source displays.

## 🔄 How It Works

The RAG workflow follows these stages:

```text
                  DOCUMENT INGESTION

      PDF Upload
          │
          ▼
     Text Extraction
          │
          ▼
    Text Chunking
          │
          ▼
   Embedding Generation
          │
          ▼
  Qdrant Vector Storage
          │
          │
          ▼
                   QUESTION ANSWERING

      User Question
          │
          ▼
     Input Validation
          │
          ▼
    Query Embedding
          │
          ▼
    Vector Retrieval
          │
          ▼
  Relevant Chunks + Metadata
          │
          ▼
   Context-aware Generation
          │
          ▼
   Output Validation
          │
          ▼
 Answer + Available Sources
```

The ingestion and query workflows are separated so documents can be indexed independently of individual questions.

## 🏗️ Architecture

Eddie Twin uses a `src`-based Python package structure with components organized around specific responsibilities.

The main architectural components include:

| Component | Responsibility |
|---|---|
| `factory.py` | Creates and wires application dependencies |
| Configuration | Loads and validates application settings |
| PDF loader | Extracts text from PDF documents |
| Chunking | Splits text into manageable chunks while preserving relevant metadata |
| Embeddings | Converts text into vector representations |
| Qdrant vector store | Stores and retrieves vectors and associated payloads |
| Retriever | Finds relevant document chunks |
| Query pipeline | Coordinates retrieval, validation, and answer generation |
| Generator | Uses an OpenAI language model to generate responses |
| Security components | Apply input and output defenses |
| `app.py` | Provides the Streamlit user interface |

The application factory makes dependencies explicit and supports substituting components during testing.

### Design principles

- **Separation of concerns:** ingestion, retrieval, generation, configuration, and presentation have distinct responsibilities.
- **Object-oriented design:** core responsibilities are represented through reusable classes.
- **Testability:** components can be tested independently of the deployed interface.
- **Metadata preservation:** source information is retained to support traceability.
- **Configuration-driven behavior:** models, vector storage, and retrieval settings can be configured through environment variables.
- **Defensive engineering:** validation and security controls are incorporated into the processing flow.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core application language |
| Streamlit | Interactive web interface |
| OpenAI API | Text embeddings and language-model generation |
| Qdrant | Vector storage and semantic retrieval |
| pypdf | PDF text extraction |
| NumPy | Numerical and vector-related operations |
| python-dotenv | Local environment configuration |
| pytest | Automated testing |
| setuptools | Python package building and installation |
| Git and GitHub | Version control and source hosting |
| Streamlit Community Cloud | Web deployment |

## ⚙️ Getting Started

### Prerequisites

Install or prepare the following:

- Python 3.10 or a compatible supported version.
- Git.
- An OpenAI API key with access to the configured models.
- A terminal and a code editor such as VS Code.

### 1. Clone the repository

```bash
git clone https://github.com/EddiesEdit/eddie-twin.git
cd eddie-twin
```

### 2. Create a virtual environment

On Ubuntu or WSL:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The project's `requirements.txt` includes `-e .`, which installs the local package in editable mode using the packaging configuration in `pyproject.toml`.

### 4. Configure environment variables

Copy the example configuration:

```bash
cp .env.example .env
```

Open `.env` and add your API key and configuration values.

See [Environment Configuration](#-environment-configuration) for the required settings.

### 5. Run the application

```bash
streamlit run app.py
```

Streamlit will provide a local URL, typically:

```text
http://localhost:8501
```

Open that URL in your browser.

## 🔐 Environment Configuration

Eddie Twin reads its configuration from environment variables. For local development, these values can be loaded from a `.env` file.

Example `.env` configuration:

```dotenv
OPENAI_API_KEY=your_openai_api_key
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_CHAT_MODEL=gpt-4o-mini
QDRANT_COLLECTION_NAME=eddie_twin_documents
QDRANT_PATH=storage/qdrant
TOP_K=5
```

### Configuration reference

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | Authenticates requests to the OpenAI API |
| `OPENAI_EMBEDDING_MODEL` | Selects the embedding model |
| `OPENAI_CHAT_MODEL` | Selects the generation model |
| `QDRANT_COLLECTION_NAME` | Names the Qdrant collection |
| `QDRANT_PATH` | Configures the local Qdrant storage path |
| `TOP_K` | Sets the number of retrieved results |

The embedding model must remain compatible with the vector dimensions expected by the vector store. The current application factory configures a vector size of 1536, matching `text-embedding-3-small` with its default output dimensions.

**Security:** Never commit `.env` or expose API keys in source code, logs, screenshots, or public repositories. Keep `.env.example` limited to placeholders and non-secret defaults.

## 🧪 Running Tests

Run the complete test suite from the repository root:

```bash
pytest -q
```

The latest verified local result:

```text
97 passed in 4.35s
```

Tests are important for validating the application's behavior as the ingestion pipeline, vector store, retrieval logic, and security controls evolve.

Run the test suite after changing application code or dependencies, and investigate failures before deploying changes.

## 🛡️ Security and Responsible Use

Security is an ongoing engineering concern for retrieval-augmented generation systems.

Eddie Twin incorporates security-oriented validation and defensive checks intended to help protect the application against malformed inputs, prompt-injection attempts, and unsafe generated content.

Important considerations include:

- Treat uploaded documents as untrusted input.
- Validate user queries before processing them.
- Do not assume retrieved text is trustworthy simply because it came from a document.
- Keep API credentials in environment variables or deployment secrets.
- Avoid storing raw prompts and generated answers in audit logs unnecessarily.
- Review retrieved sources when answers are important or ambiguous.
- Do not use generated responses as a substitute for independent verification in high-stakes decisions.

These protections reduce risk but cannot eliminate prompt injection, hallucinations, or other LLM-related vulnerabilities.

## ☁️ Deployment

Eddie Twin is deployed using Streamlit Community Cloud.

**[Open the live application](https://eddie-twin-6qwt6zd6rnybpkndhfjaxo.streamlit.app/)**

To deploy your own instance:

1. Fork or copy the GitHub repository.
2. Create a Streamlit Community Cloud application.
3. Select your repository, branch, and `app.py` entry point.
4. Configure the required secrets in the deployment settings.
5. Deploy and inspect the application logs if startup fails.

Example Streamlit Secrets configuration:

```toml
OPENAI_API_KEY = "your-real-openai-api-key"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_CHAT_MODEL = "gpt-4o-mini"
QDRANT_COLLECTION_NAME = "eddie_twin_documents"
QDRANT_PATH = "storage/qdrant"
TOP_K = 5
```

Replace the API key placeholder with your own secret. Do not commit this configuration containing a real key.

### Vector storage in deployment

The current configuration uses local Qdrant storage. The local database from your development machine is not automatically copied to Streamlit Community Cloud.

For a more persistent hosted deployment, configure an appropriately secured remote Qdrant instance and migrate or re-ingest your documents. Do not assume files written to a hosted application's local filesystem will persist indefinitely.

Refer to the [Streamlit Community Cloud documentation](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy) for deployment guidance.

## ⚠️ Current Limitations

Eddie Twin is an evolving engineering project, and the following considerations are important:

- **PDF-focused ingestion:** the current interface accepts PDF uploads.
- **LLM limitations:** generated answers may contain inaccuracies or omit relevant context.
- **Retrieval quality:** answer quality depends on document extraction, chunking, embedding quality, and retrieval settings.
- **Local storage:** the default Qdrant configuration is not a substitute for durable hosted database storage.
- **API costs:** embedding documents and generating answers consume API resources.
- **Security:** defensive checks are not a guarantee against every prompt-injection technique or unsafe output.

These limitations provide a practical basis for future engineering work.

## 🗺️ Roadmap

Potential next steps for the project include:

- [ ] Integrate hosted Qdrant for persistent cloud storage.
- [ ] Expand document ingestion and format support.
- [ ] Improve retrieval evaluation with a representative question-and-answer dataset.
- [ ] Evaluate chunk sizes, overlap, and retrieval parameters systematically.
- [ ] Add reranking and hybrid retrieval where evaluation demonstrates value.
- [ ] Expand automated tests for deployment and end-to-end RAG behavior.
- [ ] Improve observability while protecting sensitive user content.
- [ ] Add continuous integration for automated testing.
- [ ] Improve source presentation and retrieval transparency.
- [ ] Evaluate response quality, groundedness, and latency.

The roadmap is a set of potential improvements, not a claim that these features are already implemented.

## 👨‍💻 Author

**Endurance Ogashi**  
Data Scientist | Machine Learning Engineer | Production ML Systems | Natural Language Processing

I build practical AI and machine learning systems, with an interest in NLP, retrieval-augmented generation, intelligent agents, automation, and deploying maintainable AI applications.

- **GitHub:** [@EddiesEdit](https://github.com/EddiesEdit)
- **LinkedIn:** [Endurance Ogashi](https://www.linkedin.com/in/endurance-ogashi/)
- **Portfolio:** [enduranceogashi.cv](https://enduranceogashi.cv)
- **Live Project:** [Eddie Twin](https://eddie-twin-6qwt6zd6rnybpkndhfjaxo.streamlit.app/)

---

## 📄 License

No license has been specified in this README. Add a `LICENSE` file and update this section when you decide how the project may be used, modified, and distributed.

---

<p align="center">
  <strong>Built to explore the engineering behind personal AI knowledge systems.</strong>
</p>
