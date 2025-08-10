# ML Integration - Simple Summary

## The Big Picture

You're building a production ML trading system with **teacher-student distillation**. Here's what that means:

### Teacher-Student Distillation (Why Different Models?)

Think of it like this:
- **Teacher** = PhD professor with access to all resources (can use L2/L3 data, complex models)
- **Student** = Fast undergraduate who learns from the professor (only uses L1 data, must be super fast)

The student doesn't need to understand WHY, just needs to mimic WHAT the teacher predicts.

## Your Pipeline (7 Steps)

```
1. Teacher Model (Offline)
   - Train complex model (PyTorch, TFT, etc.) on rich data
   - Can be slow, doesn't matter
   
2. Student Model (Distillation)
   - Train simple model (LightGBM) to copy teacher's predictions
   - Only uses features available live
   
3. Export to ONNX
   - Convert student to ONNX format (fast, secure, portable)
   
4. Load in Actor
   - Actor loads ONNX model once at startup
   
5. Real-time Inference
   - Actor computes features from live data
   - Runs model (<2ms)
   - Publishes signal
   
6. Strategy Receives Signal
   - Gets probability from actor
   - Applies trading logic (thresholds, position sizing)
   
7. Execute Trades
   - Send orders to market
```

## Critical Rules (Never Break These)

### 1. Hot Path = Fast Path
- **NO** pandas/Polars (use numpy only)
- **NO** model loading during trading
- **NO** memory allocations per tick
- **USE** float32 everywhere
- **USE** pre-allocated buffers

### 2. Cold Path = Training Path
- **OK** to use pandas/Polars
- **OK** to be slow
- **OK** to use complex models

### 3. Security First
- **NEVER** use pickle files (security risk)
- **ALWAYS** use ONNX or native formats
- **ALWAYS** validate inputs

## Your Current Issues (Top 5)

### 🔴 Must Fix Now:
1. **Two classes named MLSignalActor** - Rename one
2. **XGBoost creating DMatrix every tick** - Use inplace_predict
3. **No feature validation** - Add schema checks

### 🟡 Fix Soon:
4. **Returning labels not probabilities** - Always return probabilities
5. **Mixed float32/float64** - Standardize on float32

## How Distillation Works (3 Methods)

### Method 1: Soft Labels (Simplest Concept)
```python
# Teacher says 70% probability up
# Train student to also predict 70%
```

### Method 2: Logit Regression (Best for Trees)
```python
# Teacher outputs logits (pre-sigmoid values)
# Student learns to match logits
# Apply sigmoid at inference
```

### Method 3: Joint Loss (Most Robust)
```python
# 50% learn from teacher
# 50% learn from real labels
# Balances teacher bias with ground truth
```

## Your Action Plan (4 Weeks)

### Week 1: Fix Critical Bugs
- Rename duplicate classes
- Fix XGBoost performance
- Add schema validation

### Week 2: Build Pipeline
- Implement teacher (complex model)
- Implement student (LightGBM)
- Export to ONNX

### Week 3: Make Production Ready
- Standardize float32
- Add monitoring
- Validate feature parity

### Week 4: Advanced Features
- Add calibration
- Multi-model support
- A/B testing

## Key Insights

1. **Teacher-Student Works Because**: The student only needs features available live, but gets the benefit of the teacher's rich training

2. **ONNX is Critical Because**: It's fast, secure, and works everywhere

3. **Float32 Matters Because**: ONNX is optimized for it, uses half the memory

4. **Feature Parity is Everything**: If training and inference features differ even slightly, predictions are garbage

5. **Probabilities Not Labels**: Trading strategies need confidence scores, not binary decisions

## Quick Wins

1. **Today**: Fix the duplicate MLSignalActor names
2. **Tomorrow**: Switch XGBoost to inplace_predict
3. **This Week**: Add schema validation
4. **Next Week**: Implement basic teacher-student pipeline

## Remember

- **Hot path** = Real-time trading = Must be fast
- **Cold path** = Training = Can be slow
- **Teacher** = Smart but slow
- **Student** = Fast approximation
- **ONNX** = Production format
- **Float32** = Standard dtype
- **No pickle** = Security rule

---

The core idea: Train a complex teacher offline, distill to a fast student, deploy via ONNX, keep hot path fast.