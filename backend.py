# In backend.py

import os
import re
import io
import json
from operator import itemgetter # <-- NEW IMPORT
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, CSVLoader, UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, Field
from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image

MODEL_NAME = "gemini-2.5-flash-preview-05-20"
vector_store = None
FILE_LOADERS = {".pdf": PyPDFLoader, ".docx": Docx2txtLoader, ".txt": TextLoader, ".csv": CSVLoader, ".xlsx": UnstructuredExcelLoader, ".xls": UnstructuredExcelLoader}

class Candidate(BaseModel):
    name: str = Field(description="The chemical name of the molecule")
    smiles: str = Field(description="The SMILES string of the molecule")

class CandidateList(BaseModel):
    """A list of candidate molecules."""
    candidates: list[Candidate] = Field(description="A list of candidate molecules that fit the user's request")

# The functions from load_documents_from_directory to generate_ligands_real are unchanged
# ...
def load_documents_from_directory(directory_path: str) -> list[Document]:
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
                    print(f"--- Error loading file {file_path}: {e} ---")
            else:
                print(f"Skipping unsupported file type: {file_path}")
    return all_docs

def load_and_index_data(folder_path):
    global vector_store
    print(f"Starting document loading from {folder_path}...")
    documents = load_documents_from_directory(folder_path)
    if not documents:
        print("No processable documents were found.")
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
    return "\n\n".join(doc.page_content for doc in docs)

def generate_ligands_real(prompt_text, vector_store_to_use):
    if not vector_store_to_use:
        return [{"name": "Error", "smiles": "Vector store not initialized."}]
    retriever = vector_store_to_use.as_retriever(search_kwargs={"k": 5})
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0.5)
    structured_llm = llm.with_structured_output(CandidateList)
    template = """
    You are a helpful chemistry expert. Based on the provided context, answer the user's question by providing a list of promising candidate ligands.
    Do not invent molecules that are not supported by the context.
    CONTEXT:
    {context}
    QUESTION:
    {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | structured_llm
    )
    try:
        response_pydantic = rag_chain.invoke(prompt_text)
        return [c.dict() for c in response_pydantic.candidates]
    except Exception as e:
        print(f"Error during AI generation: {e}")
        return [{"name": "Error processing request", "smiles": str(e)}]

def generate_synthesis_real(prompt_text, ligand_list, vector_store_to_use):
    if not vector_store_to_use:
        return "Vector store not initialized. Please index data first."
    retriever = vector_store_to_use.as_retriever(search_kwargs={"k": 10})
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0.2)
    template = """
    You are a senior process chemist... (prompt text is unchanged)
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

def sanitize_filename(name):
    name = name.strip().replace(' ', '_')
    return re.sub(r'[\\/*?:"<>|]', '', name)

def process_smiles_to_images(data_input: str) -> list:
    print("Processing SMILES to images...")
    # ... function remains unchanged ...
    lines = data_input.strip().split('\n')
    results = []
    for line in lines:
        if not line.strip(): continue
        parts = line.strip().rsplit(',', 1)
        if len(parts) != 2: continue
        name, smiles = parts[0].strip(), parts[1].strip()
        filename = f"{sanitize_filename(name)}.png"
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None: raise ValueError("Invalid SMILES")
            img_pil = Draw.MolToImage(mol, size=(300, 300))
            bio = io.BytesIO()
            img_pil.save(bio, format='PNG')
            results.append({'name': name, 'filename': filename, 'pil_image': img_pil, 'image_bytes': bio.getvalue(), 'error': None})
        except Exception as e:
            results.append({'name': name, 'filename': filename, 'pil_image': None, 'image_bytes': None, 'error': f"Invalid SMILES: {smiles}"})
    print(f"SMILES processing complete. {len(results)} entries handled.\n")
    return results

# ==============================================================================
# MODIFIED EXPLANATION FUNCTION
# ==============================================================================
def get_explanation_for_ligand(original_prompt: str, ligand_name: str, ligand_smiles: str, vector_store_to_use):
    """
    Generates a detailed explanation for why a specific ligand was suggested.
    """
    if not vector_store_to_use:
        return "Error: Vector store is not available. Please index data first."

    print(f"Generating explanation for {ligand_name}...")
    retriever = vector_store_to_use.as_retriever(search_kwargs={"k": 7})
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0.3)
    
    template = """
    You are a senior research chemist acting as an expert reviewer.
    A junior chemist made a query: "{original_prompt}"
    Based on that query and the context below from research documents, one of the suggestions was the molecule '{ligand_name}' (SMILES: {ligand_smiles}).

    Your task is to provide a detailed, scientific explanation for why this molecule was a logical suggestion.
    - Justify the choice based ONLY on the provided context.
    - Point out specific structural features, properties, or data points mentioned in the context that make this molecule a good candidate.
    - If the context does not strongly support the suggestion, be honest about it.
    - Structure your answer clearly with paragraphs. Do not just list facts.

    CONTEXT:
    {context}

    EXPLANATION:
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    # FIX: Restructure the chain to pass a simple string to the retriever
    rag_chain = (
        {
            "context": itemgetter("question_for_retriever") | retriever | format_docs,
            "original_prompt": itemgetter("original_prompt"),
            "ligand_name": itemgetter("ligand_name"),
            "ligand_smiles": itemgetter("ligand_smiles"),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    try:
        # Create a clean, simple query string specifically for the retriever
        retriever_query = f"Explain the reasoning for suggesting the molecule {ligand_name} in the context of the prompt: {original_prompt}"
        
        response = rag_chain.invoke({
            "question_for_retriever": retriever_query, # Pass the simple string here
            "original_prompt": original_prompt,
            "ligand_name": ligand_name,
            "ligand_smiles": ligand_smiles
        })
        return response
    except Exception as e:
        print(f"Error during explanation generation: {e}")
        traceback.print_exc() # Print full traceback for debugging
        return f"An error occurred while generating the explanation: {e}"