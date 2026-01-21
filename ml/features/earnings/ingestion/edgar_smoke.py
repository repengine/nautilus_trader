#!/usr/bin/env python3
"""
Minimal EDGAR API smoke test helpers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.request import Request
from urllib.request import urlopen


@dataclass(frozen=True)
class EdgarSmokeResult:
    """
    Result of an EDGAR submissions smoke test.

    Attributes
    ----------
    url:
        URL requested.
    status:
        HTTP status code.
    cik:
        CIK returned in the payload (if present).
    filings_count:
        Count of recent filings if available.
    """

    url: str
    status: int
    cik: str | None
    filings_count: int | None


def run_edgar_smoke_test(
    *,
    cik: str,
    identity: str,
    timeout_seconds: float,
) -> EdgarSmokeResult:
    """
    Fetch a submissions payload from SEC EDGAR using a compliant User-Agent.

    Args:
        cik: 10-digit CIK for the submissions endpoint.
        identity: SEC User-Agent identity string.
        timeout_seconds: Timeout for the HTTP request.

    Returns:
        EdgarSmokeResult with status and metadata.
    """
    if not identity:
        raise ValueError("SEC identity required for EDGAR smoke test")
    normalized = _normalize_cik(cik)
    url = f"https://data.sec.gov/submissions/CIK{normalized}.json"
    request = Request(url, headers={"User-Agent": identity})
    with urlopen(request, timeout=timeout_seconds) as response:
        status = getattr(response, "status", None) or 0
        payload = json.loads(response.read().decode("utf-8"))
    cik_value = _coerce_str(payload.get("cik"))
    filings_count = _extract_filings_count(payload)
    return EdgarSmokeResult(
        url=url,
        status=int(status),
        cik=cik_value,
        filings_count=filings_count,
    )


def _normalize_cik(raw: str) -> str:
    cleaned = "".join(ch for ch in raw if ch.isdigit())
    if len(cleaned) > 10:
        raise ValueError("CIK must be 10 digits or fewer")
    return cleaned.zfill(10)


def _extract_filings_count(payload: dict[str, Any]) -> int | None:
    filings = payload.get("filings")
    if not isinstance(filings, dict):
        return None
    recent = filings.get("recent")
    if not isinstance(recent, dict):
        return None
    accession_numbers = recent.get("accessionNumber")
    if not isinstance(accession_numbers, list):
        return None
    return len(accession_numbers)


def _coerce_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


__all__ = ["EdgarSmokeResult", "run_edgar_smoke_test"]
