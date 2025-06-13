import os
import re
import io
import json
from operator import itemgetter
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
import traceback

# ==============================================================================
# DEFAULT MODEL CONFIGURATION
# ==============================================================================
MODEL_NAME = "gemini-1.5-flash-latest"


# GLOBAL VARIABLES AND FILE LOADERS...
vector_store = None
FILE_LOADERS = {".pdf": PyPDFLoader, ".docx": Docx2txtLoader, ".txt": TextLoader, ".csv": CSVLoader, ".xlsx": UnstructuredExcelLoader, ".xls": UnstructuredExcelLoader}

# ==============================================================================
# PYDANTIC SCHEMAS FOR STRUCTURED OUTPUT
# ==============================================================================
class Candidate(BaseModel):
    name: str = Field(description="The chemical name of the molecule")
    smiles: str = Field(description="The SMILES string of the molecule")

class CandidateList(BaseModel):
    """A list of candidate molecules."""
    candidates: list[Candidate] = Field(description="A list of exactly five different and varied candidate molecules that fit the user's request")

class Explanation(BaseModel):
    """A detailed scientific explanation."""
    explanation_text: str = Field(description="A detailed scientific explanation for why the molecule was suggested, based on the provided context.")


# The functions from load_documents_from_directory to format_docs remain unchanged...
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
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0.7)
    
    structured_llm = llm.with_structured_output(CandidateList)
    
    template = """
    You are a helpful chemistry expert. Based on the provided context, answer the user's question.
    Your task is to provide a list of exactly five DIFFERENT and VARIED promising candidate ligands that fit the user's request.
    Each suggestion in the list must be unique. Do not suggest the same molecule multiple times.
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

def sanitize_filename(name):
    name = name.strip().replace(' ', '_')
    return re.sub(r'[\\/*?:"<>|]', '', name)

def process_smiles_to_images(data_input: str) -> list:
    print("Processing SMILES to images...")
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
    Generates a detailed explanation for why a specific ligand was suggested,
    using structured output to ensure a reliable response.
    """
    if not vector_store_to_use:
        return "Error: Vector store is not available. Please index data first."

    print(f"Generating explanation for {ligand_name}...")
    retriever = vector_store_to_use.as_retriever(search_kwargs={"k": 7})
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0.3)

    # Bind the LLM to the Explanation Pydantic schema
    structured_llm = llm.with_structured_output(Explanation)
    
    template = """
    You are a senior research chemist acting as an expert reviewer.
    Based on the provided context, provide a detailed scientific explanation for why the molecule '{ligand_name}' (SMILES: {ligand_smiles}) was a logical suggestion for the user's original query: "{original_prompt}".
    
    Justify your answer using ONLY the information found in the context below. Point out specific structural features, properties, or data points mentioned in the context.
    If the context does not strongly support the suggestion, be honest about it.

    CONTEXT:
    {context}
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    # The chain now uses the structured_llm and doesn't need a final string parser
    rag_chain = (
        {
            "context": itemgetter("question_for_retriever") | retriever | format_docs,
            "original_prompt": itemgetter("original_prompt"),
            "ligand_name": itemgetter("ligand_name"),
            "ligand_smiles": itemgetter("ligand_smiles"),
        }
        | prompt
        | structured_llm
    )
    
    try:
        # Create a clean, simple query string specifically for the retriever
        retriever_query = f"Explain the reasoning for suggesting the molecule {ligand_name} in the context of the prompt: {original_prompt}"
        
        response_pydantic = rag_chain.invoke({
            "question_for_retriever": retriever_query,
            "original_prompt": original_prompt,
            "ligand_name": ligand_name,
            "ligand_smiles": ligand_smiles
        })

        # Extract the text from the Pydantic object
        return response_pydantic.explanation_text
    except Exception as e:
        print(f"Error during explanation generation: {e}")
        traceback.print_exc()
        return f"An error occurred while generating the explanation: {e}"