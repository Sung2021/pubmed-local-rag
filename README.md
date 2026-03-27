# pubmed-local-rag

Query PubMed abstracts using a fully local LLM (Ollama) and a RAG pipeline — no API keys required.

## Requirements
- [Ollama](https://ollama.com) installed and running
- `llama3.2` model pulled (`ollama pull llama3.2`)

## Setup
```bash
pip install -r requirements.txt
```

## Usage

### 1. Build vector DB (run once per topic)
```bash
python build_db.py
```
Prompts for:
- Search term (default: `single cell RNA-seq cancer`)
- Number of papers to fetch (default: 10, max: 10000)

DB is saved to `./chroma_db/<search_term>/` — each search term gets its own folder.

### 2. Ask questions
```bash
python ask.py
```
Prompts for the DB path to load, then enters an interactive Q&A loop. Type `q` to quit.

### 3. Manage databases
```bash
python manage_db.py
```
Lists all stored databases with size and date. Supports deleting individual or all databases.

## Example Output

```
$ python ask.py
Enter DB path to use (default: ./chroma_db): ./chroma_db/single_cell_RNA_seq_cancer
Loading DB from: ./chroma_db/single_cell_RNA_seq_cancer

Enter your question (quit: q): What are the key findings about tumor microenvironment?

Answer: Several studies identified distinct immune cell populations within the tumor
microenvironment using scRNA-seq, including exhausted T cells, tumor-associated
macrophages (TAMs), and regulatory T cells. These findings suggest that the
composition of the TME is strongly associated with patient prognosis and response
to immunotherapy.
[14.2s]

Enter your question (quit: q): q
```

## Benchmark

Run `benchmark.py` to compare embedding models and LLMs across a fixed question set.
Results are saved to `benchmark_results.csv`.

```bash
ollama pull mistral  # download additional model (~4GB)
python benchmark.py
```

Models tested:

| Embedding Model | LLM | Avg Response Time |
|----------------|-----|------------------|
| all-MiniLM-L6-v2 | llama3.2 | ~14s |
| all-MiniLM-L6-v2 | mistral | ~18s |
| all-mpnet-base-v2 | llama3.2 | ~16s |
| all-mpnet-base-v2 | mistral | ~21s |

## Project Structure

```
├── build_db.py       # Fetch PubMed abstracts and build vector DB
├── ask.py            # Interactive Q&A against stored DB
├── manage_db.py      # List and delete stored databases
├── benchmark.py      # Compare embedding models and LLMs
├── requirements.txt
└── README.md
```

## Stack
- PubMed API (Biopython)
- LangChain LCEL
- ChromaDB
- HuggingFace Embeddings (`all-MiniLM-L6-v2`, `all-mpnet-base-v2`)
- Ollama (`llama3.2`, `mistral`)
