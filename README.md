# AI Chemical Discovery Assistant

A desktop application leveraging Retrieval-Augmented Generation (RAG) and Large Language Models to accelerate chemical discovery by analyzing local research data.

*(Screenshot of the application in action)*

## Overview

The AI Chemical Discovery Assistant is designed to bridge the gap between a researcher's local collection of documents (research papers, experimental data) and the powerful generative capabilities of modern AI. Instead of relying on generic knowledge, this tool builds a specialized knowledge base from your own files, allowing you to ask complex scientific questions and receive contextually aware, relevant answers.

The primary workflow involves two stages:

1.  **Ligand Discovery:** Proposing new candidate molecules based on constraints and information found in your documents.
2.  **Synthesis Planning:** Devising a detailed synthesis recipe for the molecules you have approved.

## Features

  * **Local Data Indexing:** Ingests and processes a variety of file formats from your local computer, including `.pdf`, `.docx`, `.txt`, `.csv`, and `.xlsx`.
  * **Retrieval-Augmented Generation (RAG):** Uses a local vector store (ChromaDB) to find the most relevant information from your documents before sending a query to the AI, ensuring highly relevant and factual responses.
  * **Interactive UI:** A clean, three-panel interface built with `customtkinter` that allows for a seamless workflow.
  * **Human-in-the-Loop Review:** Interactively review AI-generated ligand candidates with "Approve," "Reject," and "Feedback" controls.
  * **Two-Stage Workflow:** A dedicated process for first discovering candidates and then planning their synthesis based on your approved selections.
  * **Powered by Gemini:** Leverages Google's Gemini family of models for state-of-the-art text generation and analysis.

## Technology Stack

  * **Language:** Python 3
  * **GUI Framework:** `customtkinter`
  * **AI Framework:** `LangChain`
  * **AI Model:** Google Gemini (`gemini-1.5-flash-latest`, `embedding-001`)
  * **Vector Store:** `ChromaDB` (Local)
  * **Data Processing:** `pandas`, `pypdf`, `python-docx`, `unstructured`

## Prerequisites

Before you begin, ensure you have the following:

1.  **Python 3.8+** installed on your system.
2.  A **Google AI API Key**. You can get one from the [Google AI Studio](https://aistudio.google.com/app/apikey).

## Installation & Setup

Follow these steps to get the application running on your local machine.

1.  **Clone or Download the Repository**

      * Place the project files (`app.py`, `backend.py`, `requirements.txt`) into a folder, for example, `ai_chemical_app`.

2.  **Create and Activate a Virtual Environment**

      * Open a terminal or command prompt, navigate to your project folder, and run:

    <!-- end list -->

    ```bash
    # Create the virtual environment
    python -m venv venv

    # Activate it (Windows)
    venv\Scripts\activate

    # Activate it (macOS/Linux)
    source venv/bin/activate
    ```

3.  **Set Your Google AI API Key**

      * This is a critical step. The application reads the key from your system's environment variables for security.

      * **On Windows (Command Prompt):**

        ```cmd
        setx GOOGLE_API_KEY "PASTE_YOUR_API_KEY_HERE"
        ```

        *(Important: You must close and reopen the terminal for this change to take effect.)*

      * **On macOS/Linux (Terminal):**

        ```bash
        export GOOGLE_API_KEY="PASTE_YOUR_API_KEY_HERE"
        ```

        *(Note: To make this permanent, add the line to your `~/.bashrc` or `~/.zshrc` file.)*

4.  **Install Dependencies**

      * With your virtual environment activated, install all the required libraries from the `requirements.txt` file.

    <!-- end list -->

    ```bash
    pip install -r requirements.txt
    ```

## How to Use

1.  **Run the Application**

      * From your terminal (with the virtual environment activated), run:

    <!-- end list -->

    ```bash
    python app.py
    ```

2.  **Step 1: Index Your Data**

      * In the "Data Control" panel on the left, click **"Select Data Folder"** and choose the folder containing your research files.
      * Click the **"Index Data"** button. The application will process all supported files. This may take some time depending on the number and size of your documents. You will see progress in the terminal.
      * The status will change to "Indexing Complete\!" when finished.

3.  **Step 2: Generate Ligand Candidates**

      * In the center panel, write a detailed prompt in the "Ligand Design Prompt" box. Be specific about what you're looking for.
      * Click **"Generate Candidates"**. The AI will use the indexed data to generate a list of suggestions.
      * Review each "Ligand Card" that appears. Use the **"Approve"**, **"Reject"**, and **"Feedback"** buttons.

4.  **Step 3: Generate a Synthesis Plan**

      * As you approve ligands, their names will appear in the "Validated Ligands" list on the right.
      * Once you have a list of approved ligands, write a prompt in the "Create a synthesis plan..." box.
      * Click **"Generate Synthesis Plan"**. The AI will generate a recipe in the text box below based on your selected ligands and the knowledge from your documents.

## Project Structure

```
ai_chemical_app/
│
├── venv/                   # Virtual environment folder
├── app.py                  # Main application file (GUI and frontend logic)
├── backend.py              # Backend logic (file processing, RAG chains, AI calls)
├── requirements.txt        # List of Python dependencies
└── README.md               # This file
```
