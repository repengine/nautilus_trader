"""Public API for the playground 3D risk model utilities."""

from playground.risk_model.analysis import AnnualRiskProfile
from playground.risk_model.analysis import SectorDistanceReport
from playground.risk_model.analysis import SectorExposureSummary
from playground.risk_model.analysis import compute_annual_risk_profiles
from playground.risk_model.analysis import compute_sector_distance_reports
from playground.risk_model.analysis import summarize_eigenvalue_trends
from playground.risk_model.analysis import summarize_sector_exposures
from playground.risk_model.dataset import CoverageSummary
from playground.risk_model.dataset import FactorDataRequest
from playground.risk_model.dataset import SectorDataRequest
from playground.risk_model.dataset import SectorDataset
from playground.risk_model.dataset import SectorDatasetAssembler
from playground.risk_model.fetchers import FactorFetcherConfig
from playground.risk_model.fetchers import ProxySelection
from playground.risk_model.fetchers import SectorFetcherConfig
from playground.risk_model.fetchers import build_factor_fetcher
from playground.risk_model.fetchers import build_sector_fetcher
from playground.risk_model.pipeline import RiskPipelineConfig
from playground.risk_model.pipeline import RiskPipelineResult
from playground.risk_model.pipeline import run_risk_pipeline
from playground.risk_model.visualization import VisualizationPayload
from playground.risk_model.visualization import build_visualization_payload


__all__ = [
    "AnnualRiskProfile",
    "CoverageSummary",
    "FactorDataRequest",
    "FactorFetcherConfig",
    "ProxySelection",
    "RiskPipelineConfig",
    "RiskPipelineResult",
    "SectorDataRequest",
    "SectorDataset",
    "SectorDatasetAssembler",
    "SectorDistanceReport",
    "SectorExposureSummary",
    "SectorFetcherConfig",
    "VisualizationPayload",
    "build_factor_fetcher",
    "build_sector_fetcher",
    "build_visualization_payload",
    "compute_annual_risk_profiles",
    "compute_sector_distance_reports",
    "run_risk_pipeline",
    "summarize_eigenvalue_trends",
    "summarize_sector_exposures",
]
