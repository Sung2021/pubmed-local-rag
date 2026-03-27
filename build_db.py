import os
import re
import logging
from Bio import Entrez
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    filename="build_db.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

Entrez.email = os.getenv("ENTREZ_EMAIL", "your@email.com")

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

# Fetch full records with metadata (XML)
print("Fetching abstracts and metadata...")
try:
    fetch_handle = Entrez.efetch(db="pubmed", id=id_list, rettype="xml", retmode="xml")
    records = Entrez.read(fetch_handle)
    fetch_handle.close()
except Exception as e:
    logging.error(f"Abstract fetch failed: {e}")
    print(f"Error: Failed to fetch abstracts — {e}")
    exit(1)

# Parse records into Document objects with metadata
def extract_text(field):
    if isinstance(field, list):
        return " ".join(str(x) for x in field)
    return str(field) if field else ""

documents = []
for record in records["PubmedArticle"]:
    try:
        article = record["MedlineCitation"]["Article"]
        pmid    = str(record["MedlineCitation"]["PMID"])
        title   = extract_text(article.get("ArticleTitle", ""))
        journal = extract_text(article.get("Journal", {}).get("Title", ""))

        pub_date = article.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
        year = str(pub_date.get("Year", pub_date.get("MedlineDate", "")[:4]))

        abstract_texts = article.get("Abstract", {}).get("AbstractText", [])
        abstract = extract_text(abstract_texts)

        if not abstract:
            continue

        documents.append(Document(
            page_content=abstract,
            metadata={
                "pmid":    pmid,
                "title":   title,
                "journal": journal,
                "year":    year,
                "url":     f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            }
        ))
    except Exception as e:
        logging.warning(f"Failed to parse record: {e}")

print(f"Parsed {len(documents)} documents with metadata.")
logging.info(f"Parsed {len(documents)} documents.")

# Chunk — preserve metadata per chunk
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks.")
logging.info(f"Total chunks: {len(chunks)}")

# Build vector DB
print("Building vector database...")
try:
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    safe_term  = re.sub(r'[^a-zA-Z0-9_]', '_', term)
    persist_dir = f"./chroma_db/{safe_term}"

    batch_size = 50
    vectorstore = Chroma.from_documents(chunks[:batch_size], embeddings, persist_directory=persist_dir)
    remaining = chunks[batch_size:]
    for i in tqdm(range(0, len(remaining), batch_size), desc="Embedding chunks"):
        vectorstore.add_documents(remaining[i:i + batch_size])

except Exception as e:
    logging.error(f"Vector DB creation failed: {e}")
    print(f"Error: Failed to build vector DB — {e}")
    exit(1)

print(f"\nDone. Documents: {len(documents)}, Chunks: {len(chunks)}")
print(f"DB saved to: {persist_dir}")
logging.info(f"DB saved to: {persist_dir}")

