from audioop import add
from FAISS_functions import add_to_vector_store, show_vstore
from hashlib import new
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain


gpt4all_embd = GPT4AllEmbeddings()
# loader = PyPDFLoader("SIS_SES.pdf")
# pages = loader.load_and_split()


# faiss_index = FAISS.from_documents(pages, gpt4all_embd)

# faiss_index.save_local("faiss")

new_db = FAISS.load_local("faiss", gpt4all_embd)

# show_vstore(new_db)

add_to_vector_store('School_guidelines.pdf', new_db)
show_vstore(new_db)