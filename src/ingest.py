import os

from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from ingestion.chunking import chunk_documents
from langchain_community.document_loaders import PyPDFLoader


DATA_PATH = "data"

documents = []

print("\nStarting ingestion...\n")

for file in os.listdir(DATA_PATH):

    file_path = os.path.join(DATA_PATH, file)

    if file.endswith(".html"):
        print(f"Loading HTML: {file}")
        loader = UnstructuredHTMLLoader(file_path)
        documents.extend(loader.load())
        print(f"Finished {file}\n")

    elif file.endswith(".pdf"):
        print(f"Loading PDF: {file}")
        loader = PyPDFLoader(file_path)
        documents.extend(loader.load())
        print(f"Finished {file}\n")


print(f"✅ Total documents loaded: {len(documents)}\n")


# ✅ Chunking
print("\nSplitting documents into chunks...\n")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)


chunks = chunk_documents(documents)
print(f"✅ Total chunks created: {len(chunks)}")


if len(chunks) == 0:
    raise ValueError("No chunks created — check HTML files.")


# ✅ Embeddings
print("\nCreating embeddings... (wait, do NOT stop)\n")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(chunks, embeddings)

vectorstore.save_local("faiss_index")


print("\n✅ VECTOR DATABASE CREATED!")
print("You do NOT need to run ingestion again.\n")
