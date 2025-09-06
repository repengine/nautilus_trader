# ML Pipeline Docker Makefile
.PHONY: help build up down logs clean test health

# Default target
help:
	@echo "ML Pipeline Docker Commands:"
	@echo "  make build       - Build all Docker images"
	@echo "  make up          - Start all services"
	@echo "  make down        - Stop all services"
	@echo "  make logs        - View logs (use SERVICE=name for specific service)"
	@echo "  make clean       - Clean up volumes and images"
	@echo "  make test        - Run integration tests"
	@echo "  make health      - Check service health"
	@echo "  make pipeline    - Start only the ML pipeline service"
	@echo "  make backfill    - Run pipeline in backfill mode"
	@echo "  make realtime    - Run pipeline in realtime mode"
	@echo "  make migrate     - Apply SQL migrations to Postgres"
	@echo "  make ps          - Show container status"
	@echo "  make restart     - Restart a service (SERVICE=name)"
	@echo "  make nuke        - Stop stack and remove volumes (DANGEROUS)"
	@echo "  make grafana-import - Import default dashboard (requires GRAFANA_API_TOKEN)"
	@echo "  make ci-smoke     - CI migration smoke (start DB, apply migrations, verify)"

# Build all images
build:
	docker compose build

# Start all services
up:
	docker compose up -d

# Stop all services
down:
	docker compose down

# View logs
logs:
ifdef SERVICE
	docker compose logs -f $(SERVICE)
else
	docker compose logs -f ml_pipeline
endif

# Clean up everything
clean:
	docker compose down -v
	docker system prune -f

# Run tests
test:
	docker compose run --rm ml_pipeline python -m pytest ml/tests/integration/

# Check health
health:
	@echo "Checking service health..."
	@curl -s http://localhost:$${ML_PIPELINE_HOST_PORT:-8081}/health | python -m json.tool || echo "Pipeline not healthy"
	@docker compose ps

# Start only pipeline
pipeline:
	docker compose up -d postgres
	docker compose up ml_pipeline

# Run backfill mode
	# Use base compose; consider adding dev override with `-f ../docker-compose.dev.yml` when needed
backfill:
	docker compose run --rm \
		-e PIPELINE_MODE=backfill \
		-e BACKFILL_START=$(START) \
		-e BACKFILL_END=$(END) \
		ml_pipeline

# Run realtime mode
realtime:
	docker compose run --rm \
		-e PIPELINE_MODE=realtime \
		-e REALTIME_INTERVAL=60 \
		ml_pipeline

# Development mode with live code mounting
dev:
	docker compose -f docker-compose.yml -f ../docker-compose.dev.yml up -d postgres
	docker compose -f docker-compose.yml -f ../docker-compose.dev.yml up ml_pipeline

# Apply canonical SQL migrations to the running Postgres container
migrate:
	COMPOSE_FILE=docker-compose.yml \
	python -m ml.deployment.migrations --apply --compose-file docker-compose.yml

# Show container status
ps:
	docker compose ps

# Restart a specific service
restart:
	@test -n "$(SERVICE)" || (echo "Usage: make restart SERVICE=name" && exit 1)
	docker compose restart $(SERVICE)

# Remove everything including volumes (DANGEROUS)
nuke:
	@echo "This will remove containers and volumes for this stack!"
	docker compose down -v

# Import default Grafana dashboard via HTTP API
grafana-import:
	@test -n "$(GRAFANA_API_TOKEN)" || (echo "Set GRAFANA_API_TOKEN env var" && exit 1)
	@curl -sS -H "Content-Type: application/json" \
	  -H "Authorization: Bearer $(GRAFANA_API_TOKEN)" \
	  -X POST http://localhost:3000/api/dashboards/db \
	  -d @<(jq -n --argjson dash "$$(< grafana/ml_pipeline_health.json)" '{dashboard: $dash, overwrite: true, folderId: 0}') \
	  | jq -r '.status // .message // .'

# CI migration smoke
ci-smoke:
	python -m ml.deployment.ci_migration_smoke

# Production deployment

deploy:
	@echo "Deploying ML Pipeline to production..."
	@test -f .env || (echo "Error: .env file not found. Copy .env.example to .env and configure." && exit 1)
	docker compose --env-file .env up -d
	@echo "Deployment complete. Check logs with: make logs"

# Scale pipeline workers
scale:
	docker compose up -d --scale ml_pipeline=$(WORKERS)

# Backup database

backup:
	@echo "Backing up PostgreSQL database..."
	docker compose exec postgres pg_dump -U postgres nautilus > backup_$(shell date +%Y%m%d_%H%M%S).sql
	@echo "Backup complete."

# Restore database
restore:
	@test -f $(FILE) || (echo "Error: Backup file not found. Usage: make restore FILE=backup.sql" && exit 1)
	docker compose exec -T postgres psql -U postgres nautilus < $(FILE)
	@echo "Restore complete."
