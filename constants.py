import os
from langchain.document_loaders import PDFMinerLoader, TextLoader
from db import db

question = None

root = os.path.dirname(os.path.realpath(__file__))

SOURCE_DIRECTORY = f"{root}/SOURCE_DOCUMENTS"

db_all = db("all", root)
db_frankl = db("frankl", root)
db_inst = db("inst", root)

INGEST_THREADS = os.cpu_count()

DOCUMENT_MAP = {
    ".txt": TextLoader,
    ".pdf": PDFMinerLoader,
}

# Default Instructor Model
EMBEDDING_MODEL_NAME = "hkunlp/instructor-large"
