import os
import json
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, CSVLoader, UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.documents import Document

# --- GLOBAL VARIABLES ---
vector_store = None

# --- FILE LOADING CONFIGURATION ---
FILE_LOADERS = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt": TextLoader,
    ".csv": CSVLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".xls": UnstructuredExcelLoader,
}

# --- ROBUST FILE LOADING FUNCTION ---
def load_documents_from_directory(directory_path: str) -> list[Document]:
    """
    Manually walks through a directory and loads all supported files one by one.
    """
    all_docs = []
    for root, _, files in os.walk(directory_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            file_ext = os.path.splitext(file_name)[1].lower()

            if file_ext in FILE_LOADERS:
                loader_class = FILE_LOADERS[file_ext]
                print(f"Loading file: {file_path} using {loader_class.__name__}...")
                try:
                    loader = loader_class(file_path)
                    docs = loader.load()
                    all_docs.extend(docs)
                except Exception as e:
                    print(f"--------------------------------------------------")
                    print(f"Error loading file {file_path}: {e}")
                    print(f"Skipping this file.")
                    print(f"--------------------------------------------------")
            else:
                print(f"Skipping unsupported file type: {file_path}")

    return all_docs


def load_and_index_data(folder_path):
    """
    Loads data from files, splits it, creates embeddings, and indexes it in a vector store.
    """
    global vector_store
    
    print(f"Starting document loading from {folder_path}...")
    documents = load_documents_from_directory(folder_path)
    
    if not documents:
        print("No processable documents were found or loaded. Please check the files and logs.")
        return None

    print(f"Successfully loaded {len(documents)} documents.")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    
    print(f"Split into {len(docs)} text chunks.")

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    print("Creating vector store... (This may take a moment)")
    vector_store = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")
    
    print("Indexing complete!")
    return vector_store


def format_docs(docs):
    """Helper function to format retrieved documents for the prompt."""
    return "\n\n".join(doc.page_content for doc in docs)


def generate_ligands_real(prompt_text, vector_store_to_use):
    """
    Performs a RAG query to generate ligand candidates.
    Outputs a JSON object.
    """
    if not vector_store_to_use:
        return {"error": "Vector store not initialized. Please index data first."}

    retriever = vector_store_to_use.as_retriever(search_kwargs={"k": 5})
    # *** CHANGED MODEL NAME AND REMOVED DEPRECATED PARAMETER HERE ***
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-05-20", temperature=0.5)
    
    template = """
    You are a world-class medicinal chemist. Based on the following context from research papers and data, answer the user's question.
    Provide a list of 5 promising candidate ligands that fit the user's request.
    Do not invent molecules that are not supported by the context.
    
    CONTEXT:
    {context}
    
    QUESTION:
    {question}
    
    Your answer MUST be a single JSON object with a key "candidates", which is a list of objects, each with a "name" and "smiles" key.
    Example: {{"candidates": [{{"name": "Molecule Name", "smiles": "SMILES_STRING"}}]}}
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | JsonOutputParser()
    )
    
    try:
        response = rag_chain.invoke(prompt_text)
        return response.get("candidates", [])
    except Exception as e:
        print(f"Error during AI generation: {e}")
        return [{"name": "Error processing request", "smiles": str(e)}]


def generate_synthesis_real(prompt_text, ligand_list, vector_store_to_use):
    """
    Performs a RAG query to generate a synthesis plan.
    Outputs a string.
    """
    if not vector_store_to_use:
        return "Vector store not initialized. Please index data first."

    retriever = vector_store_to_use.as_retriever(search_kwargs={"k": 10})
    # *** CHANGED MODEL NAME AND REMOVED DEPRECATED PARAMETER HERE ***
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.2)
    
    template = """
    You are a senior process chemist. Your task is to devise a detailed synthesis recipe.
    Use the provided context from research papers and experimental data to create a plausible, step-by-step synthesis plan.
    The user wants to synthesize a new material using the following approved ligands: {ligands}
    
    Based on the context, answer the user's specific request.
    
    CONTEXT:
    {context}
    
    REQUEST:
    {question}
    
    Provide a clear, step-by-step synthesis recipe.
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough(), "ligands": lambda x: ligand_list}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    try:
        response = rag_chain.invoke(prompt_text)
        return response
    except Exception as e:
        print(f"Error during AI generation: {e}")
        return f"An error occurred: {e}"