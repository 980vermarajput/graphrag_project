# GraphRAG Project

Implementation of Graph-based Retrieval Augmented Generation using LlamaIndex, Groq, and ChromaDB.

## Project Structure

```
graphrag_project/
├── venv/               # Virtual environment
├── data/              # For your input documents
│   ├── doc1.pdf      # Add your PDF files here
│   ├── doc2.txt      # Add your text files here
│   └── ...           # Other supported document types
├── chroma_db/         # For vector store will be created on first run
├── storage/           # For index persistence will be created on first run
├── graph_rag.py       # Main implementation
├── run_graphrag.py    # Runner script
└── .env               # Environment variables (create this file)
```

## ⚙️ Setup Instructions for Windows

1. **Set up the environment**

```powershell
# Create virtual environment
python -m venv venv
# Activate virtual environment
.\venv\Scripts\activate

# For Git Bash users activate environment:
source venv/Scripts/activate
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Create .env file**
   Create a new file named `.env` and add your Groq API key:

```properties
GROQ_API_KEY=your-groq-api-key-here
```

4. **Prepare your data**
   Create a `data` folder and add your documents:

```powershell
mkdir data
# Add your documents (PDF, TXT, etc.) to the data folder
# Examples:
# - Research papers
# - Documentation
# - Text files
# - Knowledge base articles
```

5. **Run the application**

```bash
python run_graphrag.py
```
