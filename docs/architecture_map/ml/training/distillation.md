# Distillation Architecture

**Status:** Living Document
**Root:** `ml/training/`
**Key Classes:** `TFTTeacher`, `LightGBMStudentDistiller`

## 1. System Overview

The **Distillation** subsystem implements the "Teacher-Student" pattern (Hinton et al., 2015) adapted for financial time-series. It solves the latency-intelligence trade-off:

-   **Teacher (Intelligence):** A heavy `TFTTeacher` (Temporal Fusion Transformer) uses deep history, attention mechanisms, and future-known covariates (Calendar, Earnings) to understand complex regimes. It is **too slow** for the Hot Path.
-   **Student (Speed):** A lightweight `LightGBMStudentDistiller` (Gradient Boosted Trees) uses only current L1 market data features. It is trained to mimic the Teacher's **Soft Labels** (probabilities), not just the binary target.

## 2. The Distillation Protocol

1.  **Train Teacher:**
    -   Input: L1 + L2 + Macro + Calendar + Earnings.
    -   Target: Forward Returns (e.g., 15m horizon).
    -   Output: A calibrated probability score $P(y=1 | X_{teacher})$.

2.  **Generate Soft Labels:**
    -   Run the Teacher on the *Validation Set*.
    -   Save the logits $z_{teacher}$ (where $\sigma(z) = P$).

3.  **Train Student:**
    -   Input: L1 Only (simulating Actor environment).
    -   Objective: `logit_mse` (Minimize $(z_{student} - z_{teacher})^2$) or `soft_ce` (Cross-Entropy against soft probabilities).
    -   **Regularization:** `kd_lambda` controls the mix between learning from the Teacher vs. the hard Ground Truth.

4.  **Calibrate & Export:**
    -   The Student's raw output is often uncalibrated.
    -   **Platt Scaling:** A Logistic Regression is fitted on the Student's raw scores against true targets.
    -   **ONNX Baking:** The Platt parameters (`coef`, `intercept`) are "baked" into the ONNX graph as `Mul` + `Add` nodes before the final `Sigmoid`. This ensures the runtime Actor gets calibrated probabilities without needing extra logic.

## 3. Key Components

### A. Teacher (`tft_teacher.py`)

-   Uses `pytorch-forecasting`.
-   Handles "Static" vs "Time-Varying" vs "Known Future" inputs.
-   Supports **Streaming Training** (`fit_streaming`) to handle datasets larger than RAM.

### B. Student (`lightgbm.py`)

-   Uses `lightgbm` with custom objective functions (`_obj_hybrid`).
-   **Optimization:** Uses `onnxmltools` to convert the tree ensemble to ONNX.
-   **Graph Surgery:** Manually injects `Cast`, `Mul`, `Add`, `Sigmoid` nodes into the ONNX graph to enforce float32 output and calibration.

## 4. Data Flow

```mermaid
graph TD
    subgraph "Cold Path"
        A[Rich Data (L1/L2/Macro)] -->|Train| B[TFT Teacher]
        B -->|Predict| C[Soft Logits z_t]
        D[Lean Data (L1 Only)] -->|Train| E[LightGBM Student]
        C -.->|Target| E
        E -->|Calibrate| F[Platt Scaler]
        F -->|Inject| G[ONNX Graph]
        G --> H[Student Artifact]
    end

    subgraph "Hot Path"
        I[Live L1 Data] -->|Inference| H
        H -->|Prob| J[Signal]
    end
```
