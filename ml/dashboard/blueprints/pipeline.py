"""
Pipeline Blueprint for Dashboard API.

This module extracts pipeline-related routes from the monolithic app.py into a
modular Flask Blueprint for better maintainability and testability.

Routes:
    POST /api/pipeline/run - Trigger pipeline execution
    GET /api/pipeline/jobs - List pipeline jobs
    GET /api/pipeline/jobs/<job_id> - Get pipeline job details
    DELETE /api/pipeline/jobs/<job_id> - Purge pipeline job
    POST /api/pipeline/build-dataset - Build training dataset
    POST /api/pipeline/train-model - Train ML model
    POST /api/pipeline/run-hpo - Run hyperparameter optimization
    GET /api/pipeline/jobs/<job_id>/progress - Get job progress
    POST /api/pipeline/jobs/<job_id>/cancel - Cancel pipeline job
"""
from __future__ import annotations

from collections.abc import Callable
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

from flask import Blueprint
from flask import jsonify
from flask import request


if TYPE_CHECKING:
    from ml.dashboard.service import DashboardService


pipeline_bp = Blueprint("pipeline", __name__, url_prefix="/api/pipeline")


def _status_to_http(status: str, success_code: int = 200) -> int:
    """
    Convert status string to HTTP status code.

    Args:
        status: The status string from service response (e.g., 'success', 'not_found').
        success_code: HTTP code to return for success statuses. Defaults to 200.

    Returns:
        Appropriate HTTP status code for the given status.

    Example:
        >>> _status_to_http("success")
        200
        >>> _status_to_http("not_found")
        404
        >>> _status_to_http("unavailable")
        503
    """
    normalized = status.lower()
    if normalized in {"success", "purged"}:
        return success_code
    if normalized == "not_found":
        return 404
    if normalized == "unavailable":
        return 503
    if normalized in {"failed", "error"}:
        return 500
    return 500


def _pipeline_response_code(res: dict[str, Any]) -> int:
    """
    Determine HTTP response code for pipeline trigger responses.

    Args:
        res: Response dictionary from pipeline service.

    Returns:
        HTTP status code based on status and success fields.
    """
    status_token = str(res.get("status", "ERROR")).upper()
    if status_token == "QUEUED" and res.get("success"):
        return 202
    elif status_token == "UNAVAILABLE":
        return 503
    elif status_token == "INVALID":
        return 400
    elif res.get("success"):
        return 202
    else:
        return 500


def register_pipeline_routes(
    bp: Blueprint,
    svc: DashboardService,
    require_token: Callable[[], bool],
) -> None:
    """
    Register all pipeline routes on the blueprint.

    This function registers the 9 pipeline-related routes extracted from app.py.
    It delegates all business logic to the DashboardService while handling
    HTTP concerns (authentication, request parsing, response formatting).

    Args:
        bp: Flask Blueprint to register routes on.
        svc: DashboardService instance for business logic delegation.
        require_token: Callable that returns True if authentication succeeds.

    Example:
        >>> from flask import Blueprint
        >>> from ml.dashboard.service import DashboardService
        >>> bp = Blueprint("pipeline", __name__)
        >>> svc = DashboardService.from_config(config)
        >>> register_pipeline_routes(bp, svc, lambda: True)
    """

    @bp.post("/run")
    def pipeline_run() -> tuple[Any, int]:
        """
        Trigger pipeline execution.

        Accepts either 'pipeline_type' or legacy 'mode' field to specify
        the pipeline type. Configuration can be nested under 'config' key
        or provided as top-level fields.

        Returns:
            Tuple of (JSON response, HTTP status code).
            202 for queued, 400 for invalid, 503 for unavailable, 500 for errors.
        """
        if not require_token():
            return jsonify({"error": "unauthorized"}), 401

        payload = cast(dict[str, Any], request.get_json(silent=True) or {})
        pipeline_type_raw = payload.get("pipeline_type") or payload.get("mode") or "full"
        pipeline_type = str(pipeline_type_raw).strip() or "full"

        config_payload = payload.get("config")
        if isinstance(config_payload, Mapping):
            config_dict: Mapping[str, Any] = dict(config_payload)
        else:
            config_dict = {
                key: value
                for key, value in payload.items()
                if key not in {"pipeline_type", "mode", "config"}
            }

        res = svc.trigger_pipeline(pipeline_type, config_dict)
        status_code = _pipeline_response_code(res)
        return jsonify(res), status_code

    @bp.get("/jobs")
    def pipeline_jobs_list() -> tuple[Any, int]:
        """
        List all pipeline jobs.

        Returns:
            Tuple of (JSON response with jobs list, HTTP status code).
        """
        if not require_token():
            return jsonify({"error": "unauthorized"}), 401

        res = svc.list_pipeline_jobs()
        status = res.get("status", "error")
        code = _status_to_http(status)
        return jsonify(res), code

    @bp.get("/jobs/<job_id>")
    def pipeline_job_detail(job_id: str) -> tuple[Any, int]:
        """
        Get details of a specific pipeline job.

        Args:
            job_id: Unique identifier of the pipeline job.

        Returns:
            Tuple of (JSON response with job details, HTTP status code).
        """
        if not require_token():
            return jsonify({"error": "unauthorized"}), 401

        res = svc.get_pipeline_job(job_id)
        status = res.get("status", "error")
        code = _status_to_http(status)
        return jsonify(res), code

    @bp.delete("/jobs/<job_id>")
    def pipeline_job_purge(job_id: str) -> tuple[Any, int]:
        """
        Purge/delete a pipeline job.

        Args:
            job_id: Unique identifier of the pipeline job to purge.

        Returns:
            Tuple of (JSON response with purge result, HTTP status code).
        """
        if not require_token():
            return jsonify({"error": "unauthorized"}), 401

        res = svc.purge_pipeline_job(job_id)
        status = res.get("status", "error")
        code = _status_to_http(status)
        return jsonify(res), code

    @bp.post("/build-dataset")
    def pipelines_build_dataset() -> tuple[Any, int]:
        """
        Build training dataset via pipeline orchestration.

        Returns:
            Tuple of (JSON response, HTTP status code).
            202 for queued, 400 for invalid, 503 for unavailable.
        """
        if not require_token():
            return jsonify({"error": "unauthorized"}), 401

        payload = cast(dict[str, Any], request.get_json(silent=True) or {})
        res = svc.build_dataset_pipeline(config=payload)
        status_code = _pipeline_response_code(res)
        return jsonify(res), status_code

    @bp.post("/train-model")
    def pipelines_train_model() -> tuple[Any, int]:
        """
        Train ML model via pipeline orchestration.

        Returns:
            Tuple of (JSON response, HTTP status code).
            202 for queued, 400 for invalid, 503 for unavailable.
        """
        if not require_token():
            return jsonify({"error": "unauthorized"}), 401

        payload = cast(dict[str, Any], request.get_json(silent=True) or {})
        res = svc.train_model_pipeline(config=payload)
        status_code = _pipeline_response_code(res)
        return jsonify(res), status_code

    @bp.post("/run-hpo")
    def pipelines_run_hpo() -> tuple[Any, int]:
        """
        Run hyperparameter optimization via pipeline orchestration.

        Returns:
            Tuple of (JSON response, HTTP status code).
            202 for queued, 400 for invalid, 503 for unavailable.
        """
        if not require_token():
            return jsonify({"error": "unauthorized"}), 401

        payload = cast(dict[str, Any], request.get_json(silent=True) or {})
        res = svc.run_hpo_pipeline(config=payload)
        status_code = _pipeline_response_code(res)
        return jsonify(res), status_code

    @bp.get("/jobs/<job_id>/progress")
    def pipelines_progress(job_id: str) -> tuple[Any, int]:
        """
        Get pipeline job progress.

        Args:
            job_id: Unique identifier of the pipeline job.

        Returns:
            Tuple of (JSON response with progress info, HTTP status code).
        """
        if not require_token():
            return jsonify({"error": "unauthorized"}), 401

        res = svc.get_pipeline_progress(job_id)
        status = res.get("status", "error")
        code = _status_to_http(status)
        return jsonify(res), code

    @bp.post("/jobs/<job_id>/cancel")
    def pipelines_cancel(job_id: str) -> tuple[Any, int]:
        """
        Cancel a pipeline job.

        Args:
            job_id: Unique identifier of the pipeline job to cancel.

        Returns:
            Tuple of (JSON response with cancel result, HTTP status code).
            200 for success, 404 for not found, 503 for unavailable.
        """
        if not require_token():
            return jsonify({"error": "unauthorized"}), 401

        res = svc.cancel_pipeline_job(job_id)
        # Normalize status for case-insensitive comparison
        status = str(res.get("status", "")).upper()
        if res.get("success"):
            code = 200
        elif status == "NOT_FOUND":
            code = 404
        elif status == "UNAVAILABLE":
            code = 503
        else:
            code = 500
        return jsonify(res), code
