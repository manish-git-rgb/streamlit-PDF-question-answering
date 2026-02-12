import os
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA


#load env variables from dotenv
load_dotenv()

#setting working directory
working_dir = os.path.dirname(os.path.abspath(__file__)) #file means absolute path of current repository

#loading the model
embedding = HuggingFaceEmbeddings()

#loading llm
llm = ChatGroq(
    model = "llama-3.1-8b-instant",
    temperature = 0.0,
)

#document ingestion
def process_document_to_chroma_db(file_name):
# loading pdf document using unstructured file loader
    loader = UnstructuredFileLoader(f"{working_dir}/{file_name}")
    documents = loader.load()
#split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1200,
        chunk_overlap = 300
    )
    text_split = text_splitter.split_documents(documents)
#store the document chunks in chroma vector database
    vector_db = Chroma.from_documents(
        documents = text_split,
        embedding = embedding,
        persist_directory= f"{working_dir}/doc_vectorstore"
    )
    return  0

def answer_question(user_question):
#loading the persistent chroma vector database
    vectordb = Chroma(
        persist_directory= f"{working_dir}/doc_vectorstore",
        embedding_function=embedding
    )

#create a retriever for document search
    retriever = vectordb.as_retriever()

#create a retrievalQA chain to answer user question using llm
    qa_chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = "stuff",
        retriever = retriever
    )

    response = qa_chain.invoke({"query": user_question})
    answer = response["result"]

    return answer

