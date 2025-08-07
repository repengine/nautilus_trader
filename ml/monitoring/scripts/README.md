# Grafana Dashboard Management Scripts

This directory contains scripts for managing Grafana dashboards in the Nautilus Trader ML monitoring system.

## Scripts Overview

### Core Scripts

1. **`export_dashboards.py`** - Export dashboards from running Grafana instance
2. **`import_dashboards.py`** - Import dashboards to running Grafana instance
3. **`validate_dashboards.py`** - Validate dashboard JSON files
4. **`validate_config.py`** - Validate system configuration
5. **`test_integration.py`** - Integration tests for the entire system

## Quick Start

### 1. Setup Environment

```bash
# Copy example configuration
cp .env.example .env

# Edit configuration with your settings
vim .env
```

### 2. Validate Configuration

```bash
# Check configuration
python scripts/validate_config.py --detailed

# Test connectivity
python scripts/validate_config.py --check-connectivity
```

### 3. Validate Dashboards

```bash
# Validate all dashboard files
python scripts/validate_dashboards.py --input grafana/dashboards --detailed

# Validate specific dashboard
python scripts/validate_dashboards.py --file grafana/dashboards/ml-overview.json
```

### 4. Start Monitoring Stack

```bash
# Start with Docker Compose
docker-compose up -d

# Verify services are running
docker-compose ps
```

### 5. Import Dashboards

```bash
# Import all ML dashboards
python scripts/import_dashboards.py \
  --url http://localhost:3000 \
  --token YOUR_API_TOKEN \
  --input grafana/dashboards \
  --setup-folder

# Import specific dashboard
python scripts/import_dashboards.py \
  --url http://localhost:3000 \
  --token YOUR_API_TOKEN \
  --file grafana/dashboards/ml-overview.json
```

## Script Details

### export_dashboards.py

Export dashboards from a running Grafana instance.

```bash
# Export all ML dashboards
python scripts/export_dashboards.py \
  --url http://localhost:3000 \
  --token YOUR_API_TOKEN \
  --all-ml

# Export by tag
python scripts/export_dashboards.py \
  --url http://localhost:3000 \
  --token YOUR_API_TOKEN \
  --tag ml-monitoring

# Export specific dashboard
python scripts/export_dashboards.py \
  --url http://localhost:3000 \
  --token YOUR_API_TOKEN \
  --uid ml-overview
```

**Options:**

- `--url`: Grafana server URL
- `--token`: API token for authentication
- `--output`: Output directory (default: `./exported_dashboards`)
- `--uid`: Export specific dashboard by UID
- `--tag`: Export dashboards by tag
- `--all-ml`: Export all known ML dashboards
- `--verify-ssl`: Verify SSL certificates
- `--timeout`: Request timeout in seconds

### import_dashboards.py

Import dashboards to a running Grafana instance.

```bash
# Import from directory
python scripts/import_dashboards.py \
  --url http://localhost:3000 \
  --token YOUR_API_TOKEN \
  --input ./dashboards \
  --setup-folder

# Import single file
python scripts/import_dashboards.py \
  --url http://localhost:3000 \
  --token YOUR_API_TOKEN \
  --file dashboard.json
```

**Options:**

- `--url`: Grafana server URL
- `--token`: API token for authentication
- `--input`: Input directory containing JSON files
- `--file`: Import specific dashboard file
- `--folder-id`: Target folder ID
- `--setup-folder`: Auto-create ML Monitoring folder
- `--no-overwrite`: Don't overwrite existing dashboards
- `--no-validate`: Skip dashboard validation

### validate_dashboards.py

Validate dashboard JSON files for correctness and ML monitoring standards.

```bash
# Validate all dashboards in directory
python scripts/validate_dashboards.py --input ./dashboards

# Validate with detailed output
python scripts/validate_dashboards.py --input ./dashboards --detailed

# Strict mode (warnings as errors)
python scripts/validate_dashboards.py --input ./dashboards --strict

# Validate single file
python scripts/validate_dashboards.py --file dashboard.json
```

**Validation Checks:**

- JSON structure and syntax
- Required dashboard fields
- Panel configuration
- PromQL query syntax
- ML monitoring conventions
- Template variables
- Grid layout constraints
- Performance considerations

### validate_config.py

Validate system configuration and prerequisites.

```bash
# Validate default configuration
python scripts/validate_config.py

# Validate custom .env file
python scripts/validate_config.py --env-file /path/to/custom.env

# Test connectivity to services
python scripts/validate_config.py --check-connectivity

# Detailed output
python scripts/validate_config.py --detailed
```

**Configuration Sections:**

- Grafana connection settings
- Prometheus configuration
- Alertmanager setup
- Dashboard management options
- Logging configuration
- Performance settings
- Security settings

### test_integration.py

Comprehensive integration testing for the dashboard system.

```bash
# Run all tests
python scripts/test_integration.py --all

# Test specific components
python scripts/test_integration.py --test-validation
python scripts/test_integration.py --test-factory
python scripts/test_integration.py --test-files

# Test with live Grafana
python scripts/test_integration.py --test-live \
  --grafana-url http://localhost:3000 \
  --api-token YOUR_TOKEN

# Performance testing
python scripts/test_integration.py --test-performance
```

**Test Categories:**

- Dashboard factory functionality
- Dashboard validation
- Configuration validation
- Dashboard file integrity
- Live Grafana connectivity
- Dashboard rendering performance

## Authentication Methods

### API Token (Recommended)

1. Create API token in Grafana:
   - Go to Configuration > API Keys
   - Click "New API Key"
   - Set name: "ML Monitoring Dashboard Management"
   - Set role: "Admin" or "Editor"
   - Copy the generated token

2. Set in environment:

   ```bash
   export GRAFANA_API_TOKEN="your-api-token-here"
   ```

### Username/Password

```bash
export GRAFANA_USERNAME="admin"
export GRAFANA_PASSWORD="your-password"
```

## Common Workflows

### Initial Setup

1. Validate configuration
2. Start monitoring stack
3. Import dashboards
4. Verify dashboard functionality

### Regular Maintenance

1. Export dashboards for backup
2. Validate dashboard integrity
3. Update dashboards as needed
4. Monitor dashboard performance

### Development Workflow

1. Create/modify dashboard JSON
2. Validate locally
3. Test with integration tests
4. Import to development Grafana
5. Export production-ready version

## Troubleshooting

### Connection Issues

```bash
# Test basic connectivity
curl -H "Authorization: Bearer $GRAFANA_API_TOKEN" \
  http://localhost:3000/api/health

# Check service status
docker-compose ps
docker-compose logs grafana
```

### Dashboard Issues

```bash
# Validate dashboard files
python scripts/validate_dashboards.py --input ./dashboards --detailed

# Check Grafana logs
docker-compose logs grafana | grep -i error
```

### Configuration Problems

```bash
# Validate complete configuration
python scripts/validate_config.py --detailed --check-connectivity

# Check environment variables
printenv | grep GRAFANA
```

## Environment Variables

Key environment variables (see `.env.example` for complete list):

```bash
# Required
GRAFANA_URL=http://localhost:3000
GRAFANA_API_TOKEN=your-token-here

# Optional
PROMETHEUS_URL=http://localhost:9090
ML_DASHBOARD_FOLDER="ML Monitoring"
DEFAULT_DASHBOARD_REFRESH=30s
LOG_LEVEL=INFO
```

## File Structure

```
scripts/
├── README.md                 # This file
├── export_dashboards.py      # Export dashboards from Grafana
├── import_dashboards.py      # Import dashboards to Grafana
├── validate_dashboards.py    # Validate dashboard JSON files
├── validate_config.py        # Validate system configuration
└── test_integration.py       # Integration tests
```

## Dependencies

The scripts require these Python packages (see `requirements.txt`):

- `requests` - HTTP client for Grafana API
- `urllib3` - HTTP connection handling

All scripts are designed to work with Python 3.11+ and include comprehensive error handling and logging.
