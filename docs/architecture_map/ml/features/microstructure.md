# Microstructure Features

**Status:** Living Document
**Root:** `ml/features/microstructure.py`
**Key Class:** `L2MicrostructureFeatures`

## 1. System Overview

This module computes advanced features from L2 (Market Depth) and L3 (Trade Flow) data. These features capture "Order Book Pressure" and "Liquidity Dynamics" that are invisible in standard OHLCV bars.

**Key Insight:**
While price movements are the *effect*, order flow and liquidity shifts are often the *cause*.

## 2. Feature Groups

### A. Order Book Imbalance (`compute_imbalance_features`)

-   **Basic Imbalance:** $(V_{bid} - V_{ask}) / (V_{bid} + V_{ask})$.
-   **Weighted Imbalance:** Weights volume by proximity to the mid-price (orders closer to spread matter more).
-   **Multi-Level:** Imbalance at L1, Top-5, and Full Book.

### B. Market Depth (`compute_depth_features`)

-   **Total Depth:** Sum of volume on Bid vs Ask side.
-   **Depth Slope:** How quickly liquidity accumulates as you move away from the spread.
-   **VWAP Spread:** Difference between Bid-VWAP and Ask-VWAP.

### C. Spread Dynamics (`compute_spread_features`)

-   **Effective Spread:** Estimate of the actual cost to trade (taking into account walking the book).
-   **Spread Volatility:** Variance in the bid-ask gap.

### D. Trade Flow (`L3TradeFlowFeatures`)

-   **VPIN (Volume-Synchronized Probability of Informed Trading):** Measures order flow toxicity.
-   **Kyle's Lambda:** A measure of price impact (how much price moves per unit of volume traded).
-   **Trade Clustering:** Do trades happen in bursts? (High clustering = urgency).

## 3. Usage Context

-   **Cold Path (Training):** Used by `TFTDatasetBuilder` via `MicrostructureAggregator` (which caches 1-minute aggregates).
-   **Hot Path (Inference):** *Currently Batch-Only*. The `MLSignalActor` does not yet have real-time access to full L2 snapshots in the `on_bar` loop due to latency constraints.
-   *Roadmap:* Implementing an efficient C++ `L2State` maintainer for the hot path.

## 4. Data Flow

```mermaid
graph TD
    A[L2 Data (Databento MBP-10)] -->|Batch| B[MicrostructureAggregator]
    B -->|1-Min Aggregates| C[L2MinuteCache (Parquet)]
    C -->|Join| D[TFTDatasetBuilder]
    D --> E[Training Model]
```
