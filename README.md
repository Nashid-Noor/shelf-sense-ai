# ShelfSense

> 🚀 **Multi-modal system for personal library intelligence**

Transform photos of your bookshelves into a smart, searchable library with automated book detection, identification, and conversational insights.

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## ✨ Features

- **📸 Book Detection** - YOLOv8-powered detection of book spines and covers from shelf photos
- **🔤 Smart OCR** - Multi-engine text extraction (EasyOCR + Tesseract) with confidence scoring
- **🎯 Book Identification** - Semantic matching against OpenLibrary & Google Books APIs
- **🔍 Hybrid Search** - Vector similarity + BM25 keyword search with reciprocal rank fusion
- **💬 Library Chat** - Ask questions about your library with grounded, citation-backed responses
- **📊 Analytics** - Diversity metrics, reading trends, and personalized recommendations
- **⚡ Production Ready** - FastAPI backend with async support, rate limiting, and monitoring

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              Client Layer                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │   Web App   │  │  Mobile App │  │   CLI Tool  │  │  REST API   │    │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘    │
└─────────┼────────────────┼────────────────┼────────────────┼────────────┘
          │                │                │                │
          └────────────────┴────────────────┴────────────────┘
                                    │
┌───────────────────────────────────┼─────────────────────────────────────┐
│                            API Gateway (nginx)                           │
│         Rate Limiting │ SSL Termination │ Load Balancing                │
└───────────────────────────────────┼─────────────────────────────────────┘
                                    │
┌───────────────────────────────────┼─────────────────────────────────────┐
│                         FastAPI Application                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │   /books    │  │  /detect    │  │   /chat     │  │ /analytics  │    │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘    │
└─────────┼────────────────┼────────────────┼────────────────┼────────────┘
          │                │                │                │
┌─────────┼────────────────┼────────────────┼────────────────┼────────────┐
│         │          Vision Pipeline        │          RAG Pipeline       │
│         │    ┌─────────────────────┐      │    ┌─────────────────────┐  │
│         │    │   YOLOv8 Detector   │      │    │   Query Expansion   │  │
│         │    │   Layout Classifier │      │    │   Hybrid Retriever  │  │
│         │    │   ROI Extraction    │      │    │   Context Builder   │  │
│         │    │   OCR Engine        │      │    │   LLM Generator     │  │
│         │    └─────────────────────┘      │    └─────────────────────┘  │
└─────────┼────────────────┼────────────────┼────────────────┼────────────┘
          │                │                │                │
┌─────────┴────────────────┴────────────────┴────────────────┴────────────┐
│                           Data Layer                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │ PostgreSQL  │  │   FAISS     │  │    Redis    │  │ File Store  │    │
│  │  (Books)    │  │  (Vectors)  │  │   (Cache)   │  │  (Images)   │    │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- NVIDIA GPU (recommended) or CPU
- Docker & Docker Compose (for containerized deployment)

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/shelfsense-ai.git
cd shelfsense-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For GPU support (NVIDIA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Copy environment file and configure
cp .env.example .env
# Edit .env with your API keys
```

### Running Locally

```bash
# Start the API server
python -m uvicorn shelfsense.api.main:app --reload

# API available at http://localhost:8000
# Documentation at http://localhost:8000/docs
```

### Docker Deployment

```bash
# Build and start all services
docker-compose up -d

# With GPU support
docker-compose up -d api postgres redis

# View logs
docker-compose logs -f api
```

## 📖 API Reference

### Book Detection

Upload an image to detect and identify books:

```bash
curl -X POST "http://localhost:8000/api/v1/detect" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@shelf_photo.jpg"
```

Response:
```json
{
  "image_id": "abc123",
  "books": [
    {
      "bbox": [100, 50, 200, 400],
      "confidence": 0.92,
      "ocr_text": "The Great Gatsby F. Scott Fitzgerald",
      "identification": {
        "title": "The Great Gatsby",
        "author": "F. Scott Fitzgerald",
        "isbn": "9780743273565",
        "confidence": 0.95
      }
    }
  ],
  "processing_time_ms": 1234
}
```

### Library Chat

Ask questions about your library:

```bash
curl -X POST "http://localhost:8000/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "What are my longest unread books?"}'
```

Response:
```json
{
  "response": "Based on your library, the longest unread books are: \"War and Peace\" by Leo Tolstoy (1,225 pages), \"Les Misérables\" by Victor Hugo (1,462 pages)...",
  "citations": [
    {"title": "War and Peace", "author": "Leo Tolstoy"}
  ],
  "conversation_id": "conv_xyz"
}
```

### Analytics

Get library insights:

```bash
curl "http://localhost:8000/api/v1/analytics/stats"
```

See the full [API Documentation](docs/API.md) for all endpoints.

## 🔧 Configuration

Key environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | Required |
| `ANTHROPIC_API_KEY` | For RAG responses | Required for chat |
| `DEVICE` | `cuda` or `cpu` | `cuda` |
| `DETECTION_MODEL` | YOLOv8 variant | `yolov8n` |
| `OCR_ENGINE` | `easyocr`, `tesseract`, or `hybrid` | `easyocr` |

See [.env.example](.env.example) for all options.

## 📊 Performance

Benchmarks on NVIDIA RTX 3090:

| Component | Metric | Value |
|-----------|--------|-------|
| Detection | mAP@50 | 0.89 |
| OCR | Character Error Rate | 4.2% |
| Identification | Hit@5 | 0.94 |
| Full Pipeline | P95 Latency | 1.2s |

Run benchmarks:
```bash
python -m shelfsense.evaluation.benchmark --dataset ./test_data
```

## 🏛️ Project Structure

```
shelfsense-ai/
├── shelfsense/
│   ├── api/              # FastAPI application
│   │   ├── routes/       # API endpoints
│   │   ├── middleware/   # CORS, rate limiting, logging
│   │   └── schemas.py    # Pydantic models
│   ├── vision/           # Detection & image processing
│   ├── ocr/              # Text extraction
│   ├── embeddings/       # Text & visual embeddings
│   ├── identification/   # Book matching
│   ├── storage/          # Vector store & database
│   ├── rag/              # Retrieval-augmented generation
│   ├── intelligence/     # Analytics & recommendations
│   └── evaluation/       # Benchmarking tools
├── tests/                # Test suite
├── docs/                 # Documentation
├── scripts/              # Deployment scripts
├── docker-compose.yml    # Container orchestration
└── Dockerfile            # Container image
```

## 🧪 Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=shelfsense --cov-report=html

# Specific module
pytest tests/test_detection.py -v
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📝 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for object detection
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) for text recognition
- [Sentence-Transformers](https://www.sbert.net/) for semantic embeddings
- [FAISS](https://github.com/facebookresearch/faiss) for vector search
- [OpenLibrary](https://openlibrary.org/) and [Google Books](https://books.google.com/) for metadata

---

<p align="center">
  Built with ❤️ for book lovers everywhere
</p>
