import re
import logging
from Bio import Entrez
from Bio import Entrez as EntrezModule
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from tqdm import tqdm

logging.basicConfig(
    filename="build_db.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

Entrez.email = "your@email.com"

term = input("Enter search term (default: single cell RNA-seq cancer): ").strip()
if not term:
    term = "single cell RNA-seq cancer"

retmax_input = input("Number of papers to fetch (default: 10, max: 10000): ").strip()
retmax = int(retmax_input) if retmax_input.isdigit() else 10

# Search PubMed
print(f"\nSearching PubMed for: '{term}'...")
logging.info(f"Search term: '{term}', retmax: {retmax}")
try:
    search_handle = Entrez.esearch(db="pubmed", term=term, retmax=retmax)
    search_results = Entrez.read(search_handle)
    search_handle.close()
except Exception as e:
    logging.error(f"PubMed search failed: {e}")
    print(f"Error: PubMed search failed — {e}")
    exit(1)

id_list = search_results["IdList"]
if not id_list:
    print("No papers found. Try a different search term.")
    logging.warning("No results returned from PubMed.")
    exit(0)
print(f"Found {len(id_list)} papers.")
logging.info(f"Found {len(id_list)} paper IDs.")

# Fetch abstracts
print("Fetching abstracts...")
try:
    fetch_handle = Entrez.efetch(db="pubmed", id=id_list, rettype="abstract", retmode="text")
    abstracts = fetch_handle.read()
    fetch_handle.close()
except Exception as e:
    logging.error(f"Abstract fetch failed: {e}")
    print(f"Error: Failed to fetch abstracts — {e}")
    exit(1)

# Chunk text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_text(abstracts)
print(f"Split into {len(chunks)} chunks.")
logging.info(f"Total chunks: {len(chunks)}")

# Build vector DB
print("Building vector database...")
try:
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    safe_term = re.sub(r'[^a-zA-Z0-9_]', '_', term)
    persist_dir = f"./chroma_db/{safe_term}"

    batch_size = 50
    all_chunks = chunks
    first_batch = all_chunks[:batch_size]
    vectorstore = Chroma.from_texts(first_batch, embeddings, persist_directory=persist_dir)

    remaining = all_chunks[batch_size:]
    for i in tqdm(range(0, len(remaining), batch_size), desc="Embedding chunks"):
        batch = remaining[i:i + batch_size]
        vectorstore.add_texts(batch)

except Exception as e:
    logging.error(f"Vector DB creation failed: {e}")
    print(f"Error: Failed to build vector DB — {e}")
    exit(1)

print(f"\nDone. Total chunks: {len(chunks)}")
print(f"DB saved to: {persist_dir}")
logging.info(f"DB saved to: {persist_dir}")
