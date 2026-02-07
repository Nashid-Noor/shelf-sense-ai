# ShelfSense AI Makefile
# Common commands for development, testing, and deployment

.PHONY: help install dev test lint format clean docker docs run

# Colors for terminal output
BLUE := \033[34m
GREEN := \033[32m
YELLOW := \033[33m
RESET := \033[0m

# Default target
help:
	@echo "$(BLUE)ShelfSense AI$(RESET) - Available commands:"
	@echo ""
	@echo "$(GREEN)Development:$(RESET)"
	@echo "  make install     Install production dependencies"
	@echo "  make dev         Install development dependencies"
	@echo "  make run         Run the development server"
	@echo "  make shell       Open Python shell with project context"
	@echo ""
	@echo "$(GREEN)Testing:$(RESET)"
	@echo "  make test        Run all tests"
	@echo "  make test-unit   Run unit tests only"
	@echo "  make test-int    Run integration tests only"
	@echo "  make test-cov    Run tests with coverage report"
	@echo "  make test-fast   Run tests excluding slow markers"
	@echo ""
	@echo "$(GREEN)Code Quality:$(RESET)"
	@echo "  make lint        Run linters (ruff, mypy)"
	@echo "  make format      Format code (black, isort)"
	@echo "  make check       Run all checks (lint + test)"
	@echo ""
	@echo "$(GREEN)Docker:$(RESET)"
	@echo "  make docker-build    Build Docker image"
	@echo "  make docker-up       Start all services"
	@echo "  make docker-down     Stop all services"
	@echo "  make docker-logs     View service logs"
	@echo "  make docker-dev      Start in development mode"
	@echo ""
	@echo "$(GREEN)Database:$(RESET)"
	@echo "  make db-migrate  Run database migrations"
	@echo "  make db-reset    Reset database"
	@echo ""
	@echo "$(GREEN)Documentation:$(RESET)"
	@echo "  make docs        Build documentation"
	@echo "  make docs-serve  Serve documentation locally"
	@echo ""
	@echo "$(GREEN)Cleanup:$(RESET)"
	@echo "  make clean       Remove build artifacts"
	@echo "  make clean-all   Remove all generated files"

# =============================================================================
# Development
# =============================================================================

install:
	pip install -r requirements.txt

dev:
	pip install -r requirements.txt -r requirements-dev.txt
	pre-commit install

run:
	uvicorn shelfsense.api.main:app --reload --host 0.0.0.0 --port 8000

shell:
	python -c "from shelfsense import *; import IPython; IPython.embed()"

# =============================================================================
# Testing
# =============================================================================

test:
	pytest tests/ -v

test-unit:
	pytest tests/unit/ -v -m "unit or not integration"

test-int:
	pytest tests/integration/ -v -m "integration"

test-cov:
	pytest tests/ --cov=shelfsense --cov-report=html --cov-report=term

test-fast:
	pytest tests/ -v -m "not slow"

test-watch:
	ptw tests/ -- -v

# =============================================================================
# Code Quality
# =============================================================================

lint:
	@echo "$(YELLOW)Running ruff...$(RESET)"
	ruff check shelfsense/ tests/
	@echo "$(YELLOW)Running mypy...$(RESET)"
	mypy shelfsense/ --ignore-missing-imports
	@echo "$(GREEN)Linting complete!$(RESET)"

format:
	@echo "$(YELLOW)Formatting with black...$(RESET)"
	black shelfsense/ tests/
	@echo "$(YELLOW)Sorting imports with isort...$(RESET)"
	isort shelfsense/ tests/
	@echo "$(GREEN)Formatting complete!$(RESET)"

check: lint test
	@echo "$(GREEN)All checks passed!$(RESET)"

security:
	bandit -r shelfsense/ -ll

# =============================================================================
# Docker
# =============================================================================

docker-build:
	docker build -t shelfsense-ai:latest .

docker-build-dev:
	docker build -t shelfsense-ai:dev --target development .

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

docker-dev:
	docker-compose --profile dev up -d

docker-clean:
	docker-compose down -v --rmi local

# =============================================================================
# Database
# =============================================================================

db-migrate:
	alembic upgrade head

db-rollback:
	alembic downgrade -1

db-reset:
	alembic downgrade base
	alembic upgrade head

db-revision:
	alembic revision --autogenerate -m "$(msg)"

# =============================================================================
# Documentation
# =============================================================================

docs:
	mkdocs build

docs-serve:
	mkdocs serve -a 0.0.0.0:8080

# =============================================================================
# Cleanup
# =============================================================================

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	rm -rf htmlcov/
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info

clean-all: clean
	rm -rf .venv/
	rm -rf data/vectors/
	rm -rf models/*.pt
	docker-compose down -v --rmi local

# =============================================================================
# Benchmarks
# =============================================================================

benchmark:
	python -m shelfsense.evaluation.benchmark

benchmark-detection:
	python -c "from shelfsense.evaluation.benchmark import run_quick_benchmark; run_quick_benchmark('detection')"

benchmark-ocr:
	python -c "from shelfsense.evaluation.benchmark import run_quick_benchmark; run_quick_benchmark('ocr')"

# =============================================================================
# Utilities
# =============================================================================

download-models:
	@echo "$(YELLOW)Downloading YOLOv8 model...$(RESET)"
	python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
	@echo "$(YELLOW)Downloading CLIP model...$(RESET)"
	python -c "from transformers import CLIPModel; CLIPModel.from_pretrained('openai/clip-vit-base-patch32')"
	@echo "$(YELLOW)Downloading Sentence-BERT model...$(RESET)"
	python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
	@echo "$(GREEN)Models downloaded!$(RESET)"

setup-nltk:
	python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Production deployment
deploy-prod:
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Health check
health:
	curl -s http://localhost:8000/health | python -m json.tool
