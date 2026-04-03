# 📄 AI Invoice Data Extractor

A Python-based tool that uses **Llama 3.2-Vision** (via Ollama) to intelligently extract structured JSON data from both digital and handwritten invoice documents.

## Features
- **Multimodal Extraction:** Automatically switches to Vision OCR if a document is scanned or handwritten.
- **Structured Output:** Maps messy invoice text into a clean JSON schema (Vendor, Date, Total, Line Items).
- **Local AI:** Runs entirely on your machine using Ollama—no API keys or data leaving your computer.

## 🛠️ Setup Instructions

 ## Prerequisites
- Install [Ollama](https://ollama.com/)
- Pull the required model:
  ```bash
  ollama pull llama3.2-vision

# Clone the repo
git clone https://github.com/utkarsh05666/ai-extractor.git
cd ai-extractor

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

## Usage
Place your documents (PDF, TXT, Images) in the /docs folder.
Run the pipeline:
python pipeline.py

