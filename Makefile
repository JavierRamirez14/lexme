.DEFAULT_GOAL := help
COMPOSE := docker compose

-include .env
POSTGRES_USER ?= lexme
POSTGRES_DB ?= lexme
DB_CONTAINER := lexme_v2-db-1
TEST_DB := lexme_test
TEST_DB_URL := postgresql://$(POSTGRES_USER):$(POSTGRES_PASSWORD)@localhost:5432/$(TEST_DB)

.PHONY: help up up-d down down-v build logs ps health embed ingest validate-checklist ask ask-resume ask-stream ask-resume-stream contract eval eval-modo2 eval-compare eval-drift eval-calibrate-export eval-calibrate-build refset-validate-bank refset-assemble refset-generate refset-review test test-integration lint fmt fe-install fe-build

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

up: ## Rebuild changed images and start the full stack detached
	$(COMPOSE) up -d --build

up-d: ## Start the full stack detached (no rebuild)
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

validate-checklist: ## Validate the vivienda checklist against the ingested corpus
	$(COMPOSE) exec api validate-checklist --vertical vivienda

ask: ## Ask Mode 1 a question (smoke test): make ask Q="tu pregunta"
	curl -s -X POST http://localhost:8000/ask \
		-H 'Content-Type: application/json' \
		-d '{"question": "$(Q)"}'

ask-resume: ## Reply to Mode 1's disambiguating question: make ask-resume T=<thread_id> A="2017-05-04"
	curl -s -X POST http://localhost:8000/ask/resume \
		-H 'Content-Type: application/json' \
		-d '{"thread_id": "$(T)", "answer": "$(A)"}'

ask-stream: ## Stream the agentic run as SSE: make ask-stream Q="tu pregunta"
	curl -sN -X POST http://localhost:8000/ask/stream \
		-H 'Content-Type: application/json' \
		-d '{"question": "$(Q)"}'

ask-resume-stream: ## Stream the resumed run as SSE: make ask-resume-stream T=<thread_id> A="2017-05-04"
	curl -sN -X POST http://localhost:8000/ask/resume/stream \
		-H 'Content-Type: application/json' \
		-d '{"thread_id": "$(T)", "answer": "$(A)"}'

contract: ## Analyze a contract (smoke test): make contract F=path/to/contrato.pdf
	curl -s -X POST http://localhost:8000/contract/analyze -F "file=@$(F)"

eval: ## Run the Mode 1 eval harness against the real system: make eval [V=vivienda] [N=3] [JUDGE=off]
	$(COMPOSE) exec api eval run --vertical $(or $(V),vivienda) $(if $(N),--repeat $(N)) \
		$(if $(filter off,$(JUDGE)),--no-judge)

eval-modo2: ## Run the Mode 2 (false-tranquility) eval harness: make eval-modo2 [V=vivienda] [N=3]
	$(COMPOSE) exec api eval run --mode modo2 --vertical $(or $(V),vivienda) $(if $(N),--repeat $(N))

eval-compare: ## Compare a run against a baseline: make eval-compare BASE=<file> RUN=<file>
	$(COMPOSE) exec api eval compare --base $(BASE) --run $(RUN)

eval-drift: ## Measure the archive's between-session drift: make eval-drift [MODE=modo1]
	$(COMPOSE) exec api eval drift build --mode $(or $(MODE),modo1)

eval-calibrate-export: ## Draw the judge rulings of a run for human review: make eval-calibrate-export RUN=<file> [SIZE=50] [SEED=0]
	$(COMPOSE) exec api eval calibrate export --vertical $(or $(V),vivienda) \
		--run $(RUN) --size $(or $(SIZE),50) --seed $(or $(SEED),0)

# Runs on the host, not in the container: it writes the vertical's calibration record,
# and the api service mounts ./verticales read-only. It needs no database or model.
eval-calibrate-build: ## Record a reviewed sample and publish it on its run: make eval-calibrate-build SAMPLE=<file> RUN=<file> BY="<name>" KIND=human DISAGREE="3,7"
	cd api && uv run eval calibrate build --vertical $(or $(V),vivienda) \
		--sample ../eval-runs/$(SAMPLE) --stamp ../eval-runs/$(RUN) \
		--reviewed-by "$(BY)" --reviewer-kind $(or $(KIND),human) --disagree "$(DISAGREE)" \
		--out ../verticales/$(or $(V),vivienda)/eval/judge-calibration.json

refset-validate-bank: ## Check the clause bank still covers the checklist [V=vivienda]
	$(COMPOSE) exec api refset validate-bank --vertical $(or $(V),vivienda)

refset-assemble: ## Assemble pending Mode 2 contracts from the clause bank recipes [V=vivienda]
	$(COMPOSE) exec api refset assemble --vertical $(or $(V),vivienda)

refset-generate: ## Generate pending Mode 1 candidates from the query seeds [V=vivienda]
	$(COMPOSE) exec api refset generate-queries --vertical $(or $(V),vivienda)

refset-review: ## List pending reference-set candidates awaiting review [V=vivienda]
	$(COMPOSE) exec api refset review list --vertical $(or $(V),vivienda)

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

fe-build: ## Type-check and build the frontend
	cd frontend && npm run build
