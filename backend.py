import os
import sys
import re
import io
import json
from operator import itemgetter
import traceback
from typing import List, Optional

# LangChain Imports
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, CSVLoader, UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from pydantic import BaseModel, Field # Using modern Pydantic v2 import

# Agent Imports
from langchain.agents import AgentExecutor, create_react_agent
from langchain_tavily import TavilySearch # Using modern Tavily import
from langchain.tools.retriever import create_retriever_tool
# from langchain import hub  # Removed to avoid SSL issues

# Chemistry Imports
from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image

# Disable LangSmith tracking to avoid SSL issues
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_ENDPOINT"] = ""
os.environ["LANGCHAIN_API_KEY"] = ""

# ==============================================================================
# DEFAULT MODEL CONFIGURATION
# ==============================================================================
MODEL_NAME = "gemini-2.5-flash"

# GLOBAL VARIABLES
vector_store = None
agent_executor = None

# FILE LOADERS...
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

class MOFData(BaseModel):
    """Structured data for a single Metal-Organic Framework."""
    mof_name: str = Field(description="The common name or identifier of the MOF (e.g., 'CALF-20', 'UiO-66').")
    metals: str = Field(description="The metal ions or clusters mentioned for this MOF (e.g., 'Zn', 'Mg2', 'Zr6').")
    ligands: str = Field(description="The organic ligands or linkers used in this MOF (e.g., '1,2,4-Triazole', 'dobpdc').")
    property_name: str = Field(description="The name of the measured property (e.g., 'CO2 Adsorption Capacity', 'Conductivity').")
    property_value: str = Field(description="The value and units of the measured property (e.g., '2.05 mmol/g', '1.2e-4 S/cm').")

class MOFDataList(BaseModel):
    """A list of MOFs found in a text document."""
    mofs: List[MOFData] = Field(description="A list of all MOFs identified in the provided text snippet.")

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

def create_local_react_prompt():
    """Create a local React prompt template without external dependencies."""
    
    template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Question: {input}
Thought:{agent_scratchpad}"""
    
    return PromptTemplate(
        input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
        template=template
    )

def initialize_agent(vector_store_to_use):
    global agent_executor
    print("Initializing AI Agent with rule-based instructions...")
    retriever = vector_store_to_use.as_retriever(search_kwargs={"k": 7})
    retriever_tool = create_retriever_tool(retriever, "local_document_search", "Searches and returns relevant information from the user's private research papers and experimental data files.")
    search_tool = TavilySearch(max_results=5, description="A search engine for finding public scientific information, including chemical properties and SMILES strings.")
    tools = [retriever_tool, search_tool]
    
    # Use local prompt template to avoid SSL issues
    print("Using local prompt template...")
    base_prompt = create_local_react_prompt()
    
    instruction_template = """
    You are an automated research assistant specializing in chemical discovery and MOF (Metal-Organic Framework) ligand design. Your primary goal is to help researchers modify existing ligands to enhance specific properties like CO2 capture capacity, conductivity, or other relevant characteristics.

    When analyzing research papers and experimental data:
    1. First, search through the provided documents to understand the current MOF structures and their properties
    2. Identify the specific ligands mentioned and their current performance
    3. Propose modifications to these ligands that could improve the target property
    4. Provide SMILES strings for your proposed modifications
    5. Explain the rationale behind your suggestions based on chemical principles

    Always provide your final answer in the following JSON format:
    {{
        "candidates": [
            {{
                "name": "Descriptive name of the modified ligand",
                "smiles": "SMILES string of the modified structure",
                "rationale": "Brief explanation of why this modification should work"
            }}
        ]
    }}

    Be creative but scientifically sound in your suggestions. Consider factors like:
    - Functional group modifications
    - Chain length adjustments
    - Substituent effects
    - Electronic properties
    - Steric considerations
    """
    
    # Combine the base prompt with instructions
    combined_template = base_prompt.template + "\n\n" + instruction_template
    prompt = PromptTemplate(
        input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
        template=combined_template
    )
    
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0.7)
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, max_iterations=15)
    print("AI Agent initialized successfully.")

# ==============================================================================
# NEW FUNCTION: EXTRACT MOF DATA
# ==============================================================================
def extract_mof_data(docs: List[Document]) -> List[dict]:
    """
    Iterates through document chunks and extracts structured MOF data using an LLM.
    """
    print("Starting MOF data extraction from documents...")
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0)
    structured_llm = llm.with_structured_output(MOFDataList)
    extraction_prompt = ChatPromptTemplate.from_template(
        """
        You are a data extraction expert specializing in chemistry literature.
        Analyze the following text from a research paper and extract all mentions of specific Metal-Organic Frameworks (MOFs).
        For each MOF you find, provide its name, the metals involved, the organic ligands, and one key property (like CO2 capacity or conductivity) if mentioned.
        If a piece of information is not present for a specific MOF, write "N/A".
        Do not invent information. Only extract what is explicitly mentioned in the text.

        TEXT TO ANALYZE:
        {text_chunk}
        """
    )
    extraction_chain = extraction_prompt | structured_llm

    all_mofs = {}
    batch_size = 5
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i+batch_size]
        combined_text = "\n\n---\n\n".join([doc.page_content for doc in batch])
        print(f"Extracting MOFs from document batch {i//batch_size + 1}...")
        try:
            results = extraction_chain.invoke({"text_chunk": combined_text})
            for mof_data in results.mofs:
                key = (mof_data.mof_name, mof_data.metals, mof_data.ligands)
                if key not in all_mofs:
                    all_mofs[key] = mof_data.dict()
        except Exception as e:
            print(f"Could not extract data from batch {i//batch_size + 1}: {e}")
            continue

    print(f"MOF data extraction complete. Found {len(all_mofs)} unique MOF entries.")
    return list(all_mofs.values())

def load_and_index_data(folder_path):
    """
    Loads data, indexes it, and then extracts structured MOF data.
    Now returns both the vector store and the extracted MOF list.
    """
    global vector_store
    print(f"Starting document loading from {folder_path}...")
    documents = load_documents_from_directory(folder_path)
    if not documents:
        return None, None
    print(f"Successfully loaded {len(documents)} documents.")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    print(f"Split into {len(docs)} text chunks.")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    print("Creating vector store... (This may take a moment)")
    vector_store = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")
    initialize_agent(vector_store)
    print("Indexing complete! Now extracting MOF data...")
    mof_database = extract_mof_data(docs)
    print("All tasks complete!")
    return vector_store, mof_database

# --- The rest of the file (generate_ligands_real, etc.) is based on your provided agent script ---
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def generate_ligands_real(prompt_text, vector_store_to_use):
    # ... (this function is unchanged)
    global agent_executor
    if not agent_executor:
        if not vector_store_to_use: return [{"name": "Error", "smiles": "Vector store not initialized."}]
        initialize_agent(vector_store_to_use)
    try:
        print("\n--- Invoking Agent ---")
        agent_response = agent_executor.invoke({"input": prompt_text})
        agent_output_text = agent_response['output']
        print(f"\n--- Agent Final Answer (Text): ---\n{agent_output_text}")
        print("\n--- Invoking Parser for Formatting ---")
        parser_llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0).with_structured_output(CandidateList)
        parser_prompt = ChatPromptTemplate.from_template(
            "Parse the following text to extract a list of exactly five chemical candidates, including their name and SMILES string.\n\nText:\n{text_to_parse}"
        )
        parser_chain = parser_prompt | parser_llm
        response_pydantic = parser_chain.invoke({"text_to_parse": agent_output_text})
        return [c.dict() for c in response_pydantic.candidates]
    except Exception as e:
        print(f"Error during AI agent execution: {e}")
        traceback.print_exc()
        return [{"name": "Error during agent execution", "smiles": str(e)}]

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
    results = []
    lines = data_input.strip().split('\n')
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
    return results

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