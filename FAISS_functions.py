from hashlib import new
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain


gpt4all_embd = GPT4AllEmbeddings()
loader = PyPDFLoader("SIS_SES.pdf")
pages = loader.load_and_split()


# faiss_index = FAISS.from_documents(pages, gpt4all_embd)
# docs = faiss_index.similarity_search("Notification center", k=1)
# for doc in docs:
#     print(str(doc.metadata["page"]) + ":", doc.page_content[:300])
# faiss_index.save_local("faiss_index")

# new_db = FAISS.load_local("faiss_index", gpt4all_embd)

# text = new_db.similarity_search("Is there secure authentication?", k =1 )

# print(f'''>>> Page content:\n\n{text[0].page_content}''')
# print(f'''>>> Metadata:{text[0].metadata}''')
# print(f'''>>> Total docs: {len(text)}''')


# adding to the vector store 

# doc2 = PyPDFLoader('ABC_School_SIS_CRS.pdf')
# pages2 = doc2.load_and_split()
# new_db.add_documents(pages2)
# new_db.save_local("faiss_index")

# display embedded documents in vector store 
def show_vstore(store) :
    vector_df = store_to_df (store)
    print(vector_df)

def store_to_df(store) :
    v_dict = store.docstore._dict
    data_rows = []
    for k in v_dict.keys():
        doc_name = v_dict[k].metadata['source'].split('/')[-1]
        # page_number = v_dict[k].metadata['page']+1
        # content = v_dict[k].page_content
        data_rows. append({"chunk_id":k, "document": doc_name})
    vector_df = pd.DataFrame(data_rows)
    return vector_df

# show_vstore(new_db)

# delete document from embeddings store

def delete_document(store, document):
    vector_df = store_to_df(store)
    chunks_list = vector_df.loc[vector_df['document'] == document]['chunk_id'].tolist()
    store.delete(chunks_list)

def refresh_model(llm, new_store):
    retriever = new_store.as_retriever()
    model = RetrievalQAWithSourcesChain.from_chain_type(llm= llm, chain_type = "stuff", retriever = retriever)
    return model

# delete_document(new_db,'ABC_School_SIS_CRS.pdf')

## Add documents to existing vector store

def add_to_vector_store(file,store):
    loader = PyPDFLoader(file)
    pages = loader.load_and_split()
    text_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap = 0)
    docs = text_splitter.split_documents(pages)

    extension = FAISS.from_documents(docs,gpt4all_embd)
    store.merge_from(extension)