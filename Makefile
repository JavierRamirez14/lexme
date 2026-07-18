.DEFAULT_GOAL := help
COMPOSE := docker compose

.PHONY: help up up-d down down-v build logs ps health embed test lint fmt fe-install

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

up: ## Start the full stack in the foreground
	$(COMPOSE) up

up-d: ## Start the full stack detached
	$(COMPOSE) up -d

down: ## Stop the stack
	$(COMPOSE) down

down-v: ## Stop the stack and remove volumes (drops DB data and model weights)
	$(COMPOSE) down -v

build: ## Build the api and frontend images
	$(COMPOSE) build

logs: ## Follow logs from all services
	$(COMPOSE) logs -f

ps: ## Show container status
	$(COMPOSE) ps

health: ## Curl the API health endpoint
	curl -s http://localhost:8000/health

embed: ## Ask TEI for an embedding (smoke test)
	curl -s http://localhost:8080/embed -X POST \
		-H 'Content-Type: application/json' \
		-d '{"inputs": "arrendamiento de vivienda"}'

test: ## Run the API test suite
	cd api && uv run pytest

lint: ## Lint the API package
	cd api && uv run ruff check .

fmt: ## Format the API package
	cd api && uv run ruff format .

fe-install: ## Install frontend dependencies
	cd frontend && npm install
