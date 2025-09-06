# ML Documentation

## Overview

This directory contains comprehensive documentation for the Nautilus Trader ML system, a production-ready machine learning infrastructure for algorithmic trading.

**System Status**: **82% Complete** - See [ROADMAP.md](ROADMAP.md) for detailed progress tracking.

## Documentation Structure

### 📋 Strategic Planning

- **[ROADMAP.md](ROADMAP.md)** - System progress tracking and development roadmap

### 🏗️ Implementation Plans

- **[implementation/ml_pipeline_plan.md](implementation/ml_pipeline_plan.md)** - Comprehensive ML data pipeline implementation (1700 lines)
- **[implementation/data_provider_plan.md](implementation/data_provider_plan.md)** - Data provider architecture implementation

### 🎯 Architecture Guides

- **[architecture/ml_integration_guide.md](architecture/ml_integration_guide.md)** - Nautilus Trader ML integration patterns
- **[architecture/teacher_student_architecture.md](architecture/teacher_student_architecture.md)** - Teacher-student distillation framework
- **[architecture/registry_architecture.md](architecture/registry_architecture.md)** - Unified model and feature registry system

### 💻 Development Standards

- **[development/CODING_STANDARDS.md](development/CODING_STANDARDS.md)** - Comprehensive coding standards and quality requirements

### 🧰 CLI Tooling

- **[tools/CLI_Tooling.md](tools/CLI_Tooling.md)** - Build runner, dataset report, and feature promotion CLI usage

### 🔬 Research & Analysis

- **[research/freqai_analysis.md](research/freqai_analysis.md)** - FreqAI integration patterns and insights

### 📚 Module Context Documentation
Detailed documentation for each ML module:

- **[context/context_data.md](context/context_data.md)** - Data pipeline and ingestion (85% complete)
- **[context/context_features.md](context/context_features.md)** - Feature engineering (95% complete)
- **[context/context_stores.md](context/context_stores.md)** - Storage layer (90% complete)
- **[context/context_training.md](context/context_training.md)** - Training infrastructure (90% complete)
- **[context/context_registry.md](context/context_registry.md)** - Registry system (95% complete)
- **[context/context_actors.md](context/context_actors.md)** - ML actors framework (95% complete)
- **[context/context_strategies.md](context/context_strategies.md)** - Trading strategies (85% complete)
- **[context/context_deployment.md](context/context_deployment.md)** - Deployment architecture (80% complete)
- **[context/context_monitoring.md](context/context_monitoring.md)** - Monitoring infrastructure (80% complete)
- **[context/context_models.md](context/context_models.md)** - Model framework (70% complete)

### 📁 Archive
Historical documents preserved for reference:

- Investigation reports
- Validation reports
- Planning documents
- Analysis reports

## Quick Start

1. **New to the ML system?** Start with [architecture/ml_integration_guide.md](architecture/ml_integration_guide.md)
2. **Implementing features?** Review [context/context_features.md](context/context_features.md) and [development/CODING_STANDARDS.md](development/CODING_STANDARDS.md)
3. **Training models?** See [architecture/teacher_student_architecture.md](architecture/teacher_student_architecture.md)
4. **Ready to deploy?** Follow [implementation/ml_pipeline_plan.md](implementation/ml_pipeline_plan.md)

## Key Architectural Components

### Core Infrastructure

- **Data Pipeline**: ParquetDataCatalog with Databento integration
- **Feature Engineering**: Hot/cold path separation with <5ms latency
- **Model Training**: Teacher-student distillation framework
- **Storage**: PostgreSQL with partitioning and three-store architecture
- **Registry**: Unified model and feature lifecycle management
- **Deployment**: Docker Compose orchestration with Prometheus/Grafana

### Production Features

- **Performance**: Sub-5ms hot path with zero-allocation design
- **Reliability**: Circuit breakers and health monitoring
- **Observability**: Comprehensive Prometheus metrics
- **Safety**: Dry-run mode and extensive validation
- **Scalability**: Distributed training and inference support

## Documentation Standards

All documentation follows these principles:

- **Single Source of Truth**: Each concept documented in one authoritative location
- **Cross-References**: All modules linked for easy navigation
- **Code Examples**: Practical implementation examples throughout
- **Current State**: Accurate reflection of actual implementation
- **Production Focus**: Emphasis on production-ready patterns

## Contributing

When adding or updating documentation:

1. Follow the structure outlined above
2. Update cross-references in related documents
3. Ensure accuracy against actual code implementation
4. Include practical code examples
5. Update ROADMAP.md if implementation status changes

## Support

For questions or issues:

- Review relevant context documentation
- Check [development/CODING_STANDARDS.md](development/CODING_STANDARDS.md) for quality requirements
- Consult [ROADMAP.md](ROADMAP.md) for implementation status

---

**Last Updated**: 2025-08-21
**Maintainer**: ML Infrastructure Team
**Status**: Production Ready (Core), Active Development (Extensions)
