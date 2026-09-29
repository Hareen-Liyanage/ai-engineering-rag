# Ask your PDF

A small retrieval-augmented question-answering app. It extracts text from a PDF, finds relevant passages with local Ollama embeddings, and asks a local Ollama chat model to answer from those passages with page citations.

## Requirements

- Python 3.10 or newer
- [Ollama](https://ollama.com/download) installed and running
- A text-based PDF (scanned image-only PDFs need OCR first)

## Run locally

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
ollama pull llama3.2:1b
ollama pull nomic-embed-text
streamlit run main.py
```

Upload a PDF and select **Index PDF**. Once indexing finishes, enter a question and select **Get answer**. Both models run locally through Ollama, so no API key is needed.

The app uses `nomic-embed-text` for retrieval and `llama3.2:1b` for answers. Retrieved source passages and their PDF page numbers are shown below each answer. If Ollama is not already running in the background, start it in another terminal with `ollama serve`.