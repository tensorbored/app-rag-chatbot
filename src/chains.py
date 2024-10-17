import streamlit as st
# from langchain_ollama import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.vectorstores import InMemoryVectorStore
# from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, trim_messages



def load_document(uploaded_files):
    ## Process uploaded PDF's
    documents=[]
    for uploaded_file in uploaded_files:
        temppdf=f"./tmp/temp.pdf"
        with open(temppdf,"wb") as file:
            file.write(uploaded_file.getvalue())
            # file_name=uploaded_file.name

        loader=PyPDFLoader(temppdf)
        docs=loader.load()
        documents.extend(docs)
    return documents

def load_url(upload_url):
    loader=WebBaseLoader(upload_url)
    documents=loader.load()
    return documents

def create_vector_embedding(documents):
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        final_documents=text_splitter.split_documents(documents)
        # embeddings=OllamaEmbeddings(model="all-minilm")
        embeddings=HuggingFaceEmbeddings()
        # os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")
        # embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # vector_store.delete(ids=uuids[-1])
        # the below deletes all the chunks from the doc1 file
        # vectors=Chroma.from_documents(final_documents,embeddings) #,persist_directory='./repository/db')
        vectors=InMemoryVectorStore.from_documents(final_documents,embeddings) #,persist_directory='./repository/db')
        return vectors

def clear_session_state_documents_vectors(st):
    if "documents" in st.session_state:
        del st.session_state["documents"]
    if "vectors" in st.session_state:
        # https://github.com/langchain-ai/langchain/discussions/9495#discussioncomment-10503820
        # for chromaDB
        # st.session_state.vectors.reset_collection()
        # for InmemoryVectorStore
        st.session_state.vectors.adelete()
        del st.session_state["vectors"]

def create_vector_db(st):
    if "documents" in st.session_state:
        with st.spinner("Creating Vector Database..."):
            if "vectors" in st.session_state:
                st.write("Vector Database is already created")
            elif "vectors" not in st.session_state:
                st.session_state.vectors=create_vector_embedding(documents=st.session_state.documents)
                st.write("Vector Database is ready")

trimmer = trim_messages(
token_counter=len,
max_tokens=10,
strategy="last",
# token_counter=llm,
include_system=True,
allow_partial=False,
start_on="human",
)