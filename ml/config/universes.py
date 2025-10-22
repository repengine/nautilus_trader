from __future__ import annotations


"""
Centralized symbol universes for ML workflows.

This module collects commonly used universes (Tier1/2/3, supplementary ETFs,
etc.) to avoid duplication across CLIs and scripts. Callers should import from
here rather than hardcoding lists locally.
"""

# Tier 1 is stratified into two sets:
# - TIER1_CORE_12: historical “core” dozen used by lightweight smoke tests.
# - TIER1_FULL_95: the full “Intelligent TFT” 95-instrument universe.
# Downstream tooling can select either list depending on footprint needs.

TIER1_CORE_12: list[str] = [
    "SPY",
    "QQQ",
    "IWM",
    "DIA",
    "VTI",
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "META",
    "TSLA",
    "AMD",
]

TIER1_FULL_95: list[str] = [
    "AAPL.XNAS",
    "ABBV.XNAS",
    "ABT.XNAS",
    "ACN.XNAS",
    "ADBE.XNAS",
    "AMAT.XNAS",
    "AMD.XNAS",
    "AMZN.XNAS",
    "AVGO.XNAS",
    "BA.XNAS",
    "BAC.XNAS",
    "BRK.XNAS",
    "C.XNAS",
    "CAT.XNAS",
    "COIN.XNAS",
    "COP.XNAS",
    "COST.XNAS",
    "CRM.XNAS",
    "CRWD.XNAS",
    "CVX.XNAS",
    "DIA.XNAS",
    "DIS.XNAS",
    "EEM.XNAS",
    "EFA.XNAS",
    "FXE.XNAS",
    "GE.XNAS",
    "GLD.XNAS",
    "GOOG.XNAS",
    "GOOGL.XNAS",
    "GS.XNAS",
    "HD.XNAS",
    "HOOD.XNAS",
    "INTC.XNAS",
    "IWM.XNAS",
    "JNJ.XNAS",
    "JPM.XNAS",
    "KO.XNAS",
    "LCID.XNAS",
    "LLY.XNAS",
    "LUV.XNAS",
    "MA.XNAS",
    "MCD.XNAS",
    "META.XNAS",
    "MMM.XNAS",
    "MRK.XNAS",
    "MRVL.XNAS",
    "MS.XNAS",
    "MSFT.XNAS",
    "MSTR.XNAS",
    "MU.XNAS",
    "NFLX.XNAS",
    "NKE.XNAS",
    "NVDA.XNAS",
    "ORCL.XNAS",
    "OXY.XNAS",
    "PEP.XNAS",
    "PFE.XNAS",
    "PG.XNAS",
    "PLTR.XNAS",
    "PYPL.XNAS",
    "QQQ.XNAS",
    "RIVN.XNAS",
    "SLB.XNAS",
    "SLV.XNAS",
    "SOFI.XNAS",
    "SPY.XNAS",
    "T.XNAS",
    "TLT.XNAS",
    "TMO.XNAS",
    "TMUS.XNAS",
    "TSLA.XNAS",
    "TSM.XNAS",
    "TXN.XNAS",
    "UBER.XNAS",
    "UNG.XNAS",
    "UNH.XNAS",
    "USO.XNAS",
    "UUP.XNAS",
    "V.XNAS",
    "VEA.XNAS",
    "VIXY.XNAS",
    "VNQ.XNAS",
    "VNQI.XNAS",
    "VTI.XNAS",
    "VWO.XNAS",
    "VZ.XNAS",
    "WBD.XNAS",
    "WFC.XNAS",
    "WMT.XNAS",
    "XLE.XNAS",
    "XLF.XNAS",
    "XLI.XNAS",
    "XLK.XNAS",
    "XLV.XNAS",
    "XOM.XNAS",
]

TIER1_DEFAULT = TIER1_FULL_95
TIER1_SYMBOL_SETS: dict[str, list[str]] = {
    "full": TIER1_FULL_95,
    "default": TIER1_DEFAULT,
    "core": TIER1_CORE_12,
    "core12": TIER1_CORE_12,
}
TIER1_CORE = TIER1_DEFAULT

SUPPLEMENTARY_ETFS: dict[str, list[str]] = {
    "sectors": [
        "XLK",
        "XLF",
        "XLV",
        "XLE",
        "XLI",
        "XLY",
        "XLP",
        "XLB",
        "XLRE",
        "XLU",
        "XLC",
    ],
    "bonds": ["SHY", "IEF", "TLT", "TIP", "LQD", "HYG", "EMB", "AGG"],
    "commodities": ["GLD", "SLV", "USO", "UNG", "DBA", "DBB", "DBC"],
}

__all__ = [
    "SUPPLEMENTARY_ETFS",
    "TIER1_CORE",
    "TIER1_CORE_12",
    "TIER1_DEFAULT",
    "TIER1_FULL_95",
    "TIER1_SYMBOL_SETS",
]
