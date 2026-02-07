# shelf-sense

Scans bookshelves and digitizes them.

Takes a photo of a shelf, detects the spines, reads the titles via OCR, and pulls metadata from Google Books/OpenLibrary. Also has a chat interface so you can ask questions about the books you own.

## Features

- **Detection**: Custom YOLOv8 model trained on book spines.
- **OCR**: Uses EasyOCR to read spine text.
- **Identification**: Fuzzy matching against book APIs to find the real book.
- **RAG**: Vector search + LLM to chat with your library content.

## Setup

1. **Install things**
   Requires Python 3.11+.
   ```bash
   pip install -r requirements.txt
   ```

2. **Env vars**
   Copy the example and fill it out. You'll need API keys for whatever LLM you use (Google/Anthropic) if you want the chat features.
   ```bash
   cp .env.example .env
   ```

3. **Run it**
   ```bash
   python -m uvicorn shelfsense.api.main:app --reload
   ```
   API is at `http://localhost:8000`.

## Docker

If you prefer docker:
```bash
docker-compose up -d
```

## Usage

Upload an image to `/api/v1/detect` to get a list of books.
Use `/api/v1/chat` to talk to your library.

## Notes

- The spine detector filters out non-book objects but might miss thin spines.
- OCR on vertical text is hit-or-miss depending on the angle.
- Data lives in postgres/pgvector.
