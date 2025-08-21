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

# Build all images
build:
	docker-compose build

# Start all services
up:
	docker-compose up -d

# Stop all services
down:
	docker-compose down

# View logs
logs:
ifdef SERVICE
	docker-compose logs -f $(SERVICE)
else
	docker-compose logs -f ml_pipeline
endif

# Clean up everything
clean:
	docker-compose down -v
	docker system prune -f

# Run tests
test:
	docker-compose run --rm ml_pipeline python -m pytest ml/tests/integration/

# Check health
health:
	@echo "Checking service health..."
	@curl -s http://localhost:8080/health | python -m json.tool || echo "Pipeline not healthy"
	@docker-compose ps

# Start only pipeline
pipeline:
	docker-compose up -d postgres
	docker-compose up ml_pipeline

# Run backfill mode
backfill:
	docker-compose run --rm \
		-e PIPELINE_MODE=backfill \
		-e BACKFILL_START=$(START) \
		-e BACKFILL_END=$(END) \
		ml_pipeline

# Run realtime mode
realtime:
	docker-compose run --rm \
		-e PIPELINE_MODE=realtime \
		-e REALTIME_INTERVAL=60 \
		ml_pipeline

# Development mode with live code mounting
dev:
	docker-compose -f docker-compose.yml -f docker-compose.override.yml up ml_pipeline

# Production deployment
deploy:
	@echo "Deploying ML Pipeline to production..."
	@test -f .env || (echo "Error: .env file not found. Copy .env.example to .env and configure." && exit 1)
	docker-compose --env-file .env up -d
	@echo "Deployment complete. Check logs with: make logs"

# Scale pipeline workers
scale:
	docker-compose up -d --scale ml_pipeline=$(WORKERS)

# Backup database
backup:
	@echo "Backing up PostgreSQL database..."
	docker-compose exec postgres pg_dump -U postgres nautilus > backup_$(shell date +%Y%m%d_%H%M%S).sql
	@echo "Backup complete."

# Restore database
restore:
	@test -f $(FILE) || (echo "Error: Backup file not found. Usage: make restore FILE=backup.sql" && exit 1)
	docker-compose exec -T postgres psql -U postgres nautilus < $(FILE)
	@echo "Restore complete."