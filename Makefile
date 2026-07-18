.DEFAULT_GOAL := help
COMPOSE := docker compose

-include .env
POSTGRES_USER ?= lexme
POSTGRES_DB ?= lexme
DB_CONTAINER := lexme_v2-db-1
TEST_DB := lexme_test
TEST_DB_URL := postgresql://$(POSTGRES_USER):$(POSTGRES_PASSWORD)@localhost:5432/$(TEST_DB)

.PHONY: help up up-d down down-v build logs ps health embed ingest test test-integration lint fmt fe-install

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

ingest: ## Build or refresh the vivienda corpus inside the api container
	$(COMPOSE) exec api ingest --manifest /verticales/vivienda/manifest.json

test: ## Run the API test suite (DB integration tests skip unless configured)
	cd api && uv run pytest

test-integration: ## Run the full suite including DB tests against a lexme_test database
	docker exec $(DB_CONTAINER) psql -U $(POSTGRES_USER) -d $(POSTGRES_DB) -tc \
		"SELECT 1 FROM pg_database WHERE datname='$(TEST_DB)'" | grep -q 1 || \
		docker exec $(DB_CONTAINER) psql -U $(POSTGRES_USER) -d $(POSTGRES_DB) -c "CREATE DATABASE $(TEST_DB)"
	cd api && LEXME_TEST_DATABASE_URL=$(TEST_DB_URL) uv run pytest

lint: ## Lint the API package
	cd api && uv run ruff check .

fmt: ## Format the API package
	cd api && uv run ruff format .

fe-install: ## Install frontend dependencies
	cd frontend && npm install
