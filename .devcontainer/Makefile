# Dev Container Makefile for convenience

.PHONY: help build up down restart logs shell exec stop clean rebuild

help:
	@echo "SQL Agent Dev Container Commands"
	@echo "=================================="
	@echo "make build       - Build the dev container"
	@echo "make up          - Start the dev container"
	@echo "make down        - Stop the dev container"
	@echo "make restart     - Restart the dev container"
	@echo "make logs        - View container logs"
	@echo "make shell       - Open shell in container"
	@echo "make clean       - Remove containers and volumes"
	@echo "make rebuild     - Rebuild container from scratch"
	@echo "make streamlit   - Run Streamlit app"
	@echo "make test        - Run tests"
	@echo "make format      - Format code with black"
	@echo "make lint        - Lint code with flake8"

build:
	docker-compose -f .devcontainer/docker-compose.yml build

up:
	docker-compose -f .devcontainer/docker-compose.yml up -d
	@echo "✅ Container started. Access at http://localhost:8501"

down:
	docker-compose -f .devcontainer/docker-compose.yml down

restart: down up

logs:
	docker-compose -f .devcontainer/docker-compose.yml logs -f dev

shell:
	docker-compose -f .devcontainer/docker-compose.yml exec dev bash

exec:
	docker-compose -f .devcontainer/docker-compose.yml exec dev $(CMD)

stop:
	docker-compose -f .devcontainer/docker-compose.yml stop

clean:
	docker-compose -f .devcontainer/docker-compose.yml down -v
	@echo "✅ Cleaned up containers and volumes"

rebuild: clean build up

streamlit:
	docker-compose -f .devcontainer/docker-compose.yml exec dev \
		streamlit run Agent-Case/SQL-Agent/app/brick_chat.py

test:
	docker-compose -f .devcontainer/docker-compose.yml exec dev pytest

format:
	docker-compose -f .devcontainer/docker-compose.yml exec dev black .

lint:
	docker-compose -f .devcontainer/docker-compose.yml exec dev flake8 .

jupyter:
	docker-compose -f .devcontainer/docker-compose.yml exec dev jupyter lab --ip=0.0.0.0

.DEFAULT_GOAL := help
