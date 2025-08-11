Here’s a grounded review and concrete recommendations to strengthen feature engi
neering, parity, and promotion flows.

**Key Findings**

- Feature engineer: Centralized, parity-minded design in `ml/features/engineering.py`; online path reuses a preallocated buffer, batch path uses Nautilus indicators and mirrors online math.
- Parity validator: `ml/features/validation.py` warms indicators and compares ba
tch vs. online with a clear tolerance. Good foundation for CI gating.
- Signal actor: `ml/actors/signal.py` wires `FeatureEngineer` and `IndicatorManager` correctly; strategies are cleanly separated and hot path is allocation-aware.
- FreqAI patterns: Feature expansion (timeframes, shifted candles, corr pairs), strong pipeline (scaler, variance filter, DI/SVM/DBSCAN/PCA), timestamped model versions, retention, and acceptance gating.

**Gaps & Risks**

- Scaling mismatch: Online path passes `scaler=None` in `_compute_features`; if training used scaling, inference may be inconsistent. Risk of degraded predictions.
- MACD completeness: `macd_signal` and `macd_diff` are zeroed (Nautilus MACD lac
ks those). Features risk being misleading/noisy.
- Hardcoded knobs: Caps and windows (e.g., price position 20, intensity 5.0, impact cap 0.01, epsilon 1e-10) are inline; should live in config/constants to honor “no hardcoding” rule.
- Microstructure/trade-flow: OHLCV fallbacks are heuristic; real L2/L3-derived features should be gated-on when available (otherwise omit to avoid spurious signal).
- Feature signature/versioning: No explicit feature signature stored with models; promotion can’t validate compatibility.
- Promotion gates: No automated acceptance criteria (latency, confidence, feature parity, accuracy delta) before activating models.

**Recommendations**

- Scaling parity
  - Persist fitted scaler with the model artifact and metadata; load and pass to
 `calculate_features_online(...)` in `MLSignalActor`.
  - Add a scaler null-guard: if absent → skip scaling; else → apply consistently.
- MACD signal/diff
  - Implement internal EMAs on MACD line for `signal` and `diff` (same math both
 paths), or drop these features; zero constants are worse than omission.
- Config hygiene
  - Move all magic numbers to `ml/config/constants.py` and/or `FeatureConfig` (DRY, centralization, SoC).
- Feature signature & versioning
  - Compute a stable hash of `FeatureConfig.get_feature_names()` + indicator specs; store in model metadata.
  - At load time, compare model’s `feature_signature` with live `FeatureConfig`; reject or warn if mismatch.
- Acceptance gates (promotion)
  - Define `ValidationCriteria` (max feature error, p99 latency, accuracy delta vs. teacher, min confidence).
  - Run `FeatureParityValidator` in CI and before promotion; fail on tolerance breach.
  - Track performance stats during shadow mode; promote only if criteria pass.
- FreqAI-style expansion (optional, staged)
  - Add configurable multi-timeframe expansion and shifted candles (batch only),
 then mirror in online via indicator state and/or precomputed informative streams.
  - If you adopt outlier removal or PCA, keep pipelines centralized (single module), persisted, and applied both train/infer.
- Microstructure/trade-flow
  - Gate by data availability; if no L2/L3 or trades, omit rather than approximate.
  - When available, add robust features (e.g., bid-ask spread percentiles, order
-imbalance, VPIN-ish metrics) with strict online/batch parity.

**Immediate Next Steps**

- Persist scaler and apply online in `MLSignalActor` (and add metadata flag “scaled=True/False”).
- Replace MACD zeros by computed `signal`/`diff` or drop them consistently in both paths.
- Extract inline constants to `ml/config/constants.py` and/or `FeatureConfig`.
- Add a `feature_signature` hash and enforce it at load/predict time.
- Wire `FeatureParityValidator.validate_parity` into tests and promotion checks; fail CI on breach.
- Optional: design a minimal, opt-in feature expansion layer (timeframes + shifts) and a simple variance-threshold step, persisted with the model.

-------------------------------------

VERSION 2:

**Executive Summary**

- Build a first-class feature system mirroring the model system: a typed “Featur
e Registry” with manifests, a deterministic pipeline (batch/online parity), life
cycle automation (create → validate → promote → deprecate → scrap), and runtime
governance (latency/footprint contracts).
- Leverage the existing FeatureEngineer and FeatureParityValidator as the hot/co
ld math source of truth; add a manifest and pipeline interfaces around them.
- Borrow FreqAI’s strengths (configurable pipelines, multi-timeframe features, c
orrelation pairs, shift windows) where they fit our latency and consistency cons
traints; maintain L1-only guarantees for students and allow L2/L3-rich teachers
offline.
- Integrate features into deployment: students load a “feature set” by ID from t
he Feature Registry, actor validates schema/hash, and uses pre-allocated online
transforms with Nautilus indicators.

**Current State (Findings)**

- Feature computation:
  - FeatureEngineer and IndicatorManager already enforce batch/online parity wit
h strict math and tolerance checks. Online path uses pre-allocated buffers; batc
h uses Polars/Pandas.
  - FeatureParityValidator warms indicators and validates end-to-end parity, sha
pes, and per-feature bounds.
  - Features include returns, volatility, RSI/Bollinger/ATR/EMA/MACD, price posi
tion, HL spread, plus optional microstructure/trade-flow with graceful OHLCV fal
lbacks.
- Actor integration (MLSignalActor):
  - Uses IndicatorManager + FeatureEngineer hot path; satisfies low-latency targ
ets with pre-allocation, optional lock-free buffers, and timing metrics.
  - Supports registry-loaded models and metadata, but features are configured vi
a FeatureConfig rather than a stable feature-set identity.
- Registry (Model Registry Docs):
  - Models have self-describing manifests (role, data requirements, feature sche
ma/hash, constraints, lineage).
  - No analogous, first-class Feature Registry yet; features are implicit in cod
e/config.

**Gaps vs. Maximum Capabilities**

- No canonical feature-set identity:
  - Training and inference depend on FeatureConfig but lack a persisted “feature
 set” manifest linking model artifacts to a specific, hashed feature schema and
pipeline version.
- Lifecycle and governance:
  - No automation to create, validate, promote, deprecate, and scrap feature set
s with quality and latency gates.
- Pipeline orchestration:
  - Transform stages exist inside FeatureEngineer, but there’s no declarative DA
G or plugin API to compose features (e.g., multi-timeframe, correlation pairs, s
hifted candles) with compile-time checks and runtime capability gating.
- Cross-artifact lineage:
  - Models reference “feature_names” in metadata, but not a stable FeatureSet ID
 with schema hash, transformation graph, and resource contracts.
- Operations:
  - Drift/quality telemetry exists partially (metrics collector hooks) but not a
 standardized per-feature drift/coverage report tied to the registry and promoti
on gates.

**Design Goals**

- Determinism and parity: single source-of-truth transforms with strict batch/on
line equivalence.
- Typed manifests and contracts: the same rigor as ModelManifest applied to feat
ure sets.
- Performance-first hot path: zero allocations, L1-only for students; teacher fe
atures can be rich offline but must distill to L1-only at serve.
- Deployment automation: manifest-driven loading and validation; fail fast if mi
smatched.

**Feature Registry**

- FeatureManifest (self-describing, versioned)
  - Identity: `feature_set_id`, `name`, `version`, `created_at`.
  - Role: `FeatureRole` (TEACHER, STUDENT, INFERENCE_SUPPORT).
  - Data Requirements: `L1_ONLY | L1_L2 | L1_L2_L3 | OfflineOnly`.
  - Schema:
    - `feature_names` (ordered list).
    - `feature_dtypes` (e.g., all float32).
    - `schema_hash` (hash(feature_names + dtypes + pipeline_signature)).
  - Pipeline:
    - `pipeline_signature` (hash of the transform graph + parameters).
    - `pipeline_version` (semantic version of the engine).
    - `capability_flags` (microstructure, trade_flow, multi_timeframe, corr_pair
s, shifts).
  - Constraints:
    - `max_latency_ms_hot_path`, `max_memory_mb_hot_path`, `max_init_warmup_bars
`.
  - Parity and Quality:
    - `parity_tolerance`, `parity_report_digest` (max diff, failing features).
    - `drift_monitors` (enabled monitors).
  - Lineage:
    - `parent_feature_set_id` (e.g., teacher feature set).
    - Links to `model_ids` trained with this feature set (reverse index in regis
try).
  - Metadata:
    - `notes`, `tags`, `owner`, `environment`.

- FeatureRegistry API (LocalFeatureRegistry analogous to LocalModelRegistry)
  - `register_feature_set(manifest: FeatureManifest, artifacts: FeatureArtifacts
) -> feature_set_id`
    - Artifacts: optional scaler parameters, pipeline config snapshot, offline s
tats.
  - `promote(feature_set_id, stage: “candidate|staging|prod”) -> None`
  - `deprecate(feature_set_id, reason) -> None`
  - `scrap(feature_set_id) -> None`
  - `get_feature_set(feature_set_id) -> FeatureManifest`
  - `resolve_by_schema_hash(schema_hash) -> feature_set_id | None`
  - `list_by_role(DataRequirements/FeatureRole)`; lineage queries.

- Sidecar files
  - `feature_set.manifest.json` (manifest above).
  - `feature_pipeline.json` (declarative config; see pipeline below).
  - `scaler.npz` (optional; training-fitted; never refit online).
  - `parity_report.json` and `perf_report.json` snapshots.

**Feature Pipeline**

- Single implementation surface with two execution modes:
  - Batch mode (cold): Polars/Pandas, vectorized but parity-preserving; uses Nau
tilus indicators iteratively if needed to ensure identical math to online path.
  - Online mode (hot): pre-allocated numpy buffers; Nautilus indicators; zero al
location; identical math.

- Declarative pipeline spec (subset of FreqAI concepts, constrained by latency)
  - Transforms (ordered, typed, parameterized):
    - price_returns(periods), momentum(periods)
    - volatility(window)
    - volume_ratio(periods)
    - indicators: rsi, bb, atr, ema_fast/slow, macd, price_position(range), hl_s
pread
    - optional: keltner, obv, book_imbalance, spread_analyzer, microstructure, t
rade_flow
    - multi_timeframe: downsampled aggregations (teacher only or gated offline)
    - correlation_pairs: teacher only; distilled to student label/knowledge, not
 served online
    - shifted_candles(n): teacher only or offline support; avoid online state ex
plosion
  - Capability gating:
    - Student feature sets must declare `L1_ONLY`; any L2/L3 transforms disabled
 at compile-time.
    - Hot path checks ensure only allowed transforms are compiled.
  - Compile-time pipeline signature:
    - Hash of the ordered transforms + params + gating decisions.
    - Combined into FeatureManifest.schema_hash.

- Plugin API
  - `FeatureTransform` interface:
    - `name: str`
    - `requires: set[Capabilities]` (e.g., L2 data)
    - `applies_to: (“batch”, “online”, “both”)`
    - `fit(X)`, `transform_batch(df)`, `transform_online(current_bar, state)`, `
stateful: bool`
  - Registration into a simple catalog (no dynamic aliasing); aligns with model
registry’s clarity:
    - `register_transform(FeatureTransform)`
    - Use explicit names in pipeline spec; no hidden aliases.

- Parity and Latency
  - Parity tests: run FeatureParityValidator as part of register/promote, store
digest in manifest.
  - Latency tests: run microbenchmarks per pipeline, store P50/P95/P99 metrics i
n manifest.

**Lifecycle Automation**

- Creation:
  - Trainer (teacher/student) declares pipeline spec and target DataRequirements
.
  - Runs batch feature build; fits scaler on training split; saves scaler and pi
peline snapshot.
  - Runs parity check on holdout; stores `parity_report.json`.
  - Runs latency benchmark; stores `perf_report.json`.
  - Computes `schema_hash` and `pipeline_signature`.
  - Registers feature set in FeatureRegistry and returns `feature_set_id`.
- Promotion:
  - Auto gates:
    - Parity max diff ≤ tolerance.
    - Hot path P99 latency within budget (<0.5 ms features, end-to-end <5 ms).
    - For students: asserts `L1_ONLY`.
  - On success, mark as “staging” or “prod”; canary rollout controlled by strate
gy and model registry.
- Deprecation and Scrapping:
  - After replacement, set `deprecated_at` and block new deployments.
  - Scrap to remove or archive artifacts; maintain lineage and audit.
- Drift & Quality:
  - Online actor emits per-feature latency, optional drift metrics (PSI/KS), NaN
 rate, saturation counts.
  - Daily job computes drift report and triggers promotion rollback if threshold
s exceeded.

**Actor and Strategy Integration**

- Loading
  - MLSignalActorConfig extended:
    - `feature_set_id: Optional[str]` and `use_registry_features: bool`
    - When provided, actor loads FeatureManifest and configures FeatureEngineer
deterministically.
    - Validates `feature_names` equality and `schema_hash` before serving.
- Execution
  - Hot path uses pre-allocated buffer matching manifest schema order; fails fas
t on mismatch.
  - Respects FeatureManifest constraints:
    - Aborts if capability-conflicting transform is present (e.g., L2 required f
or student).
- Telemetry
  - Emit:
    - `feature_computation_seconds{actor_id}` histogram
    - `feature_drift_score{feature,actor_id}` (if enabled)
    - `feature_nan_rate{feature,actor_id}`
  - Link metrics with `feature_set_id` and `model_id` labels for registry backfi
ll.

**FreqAI Inspiration (Adopt With Discipline)**

- What to adopt:
  - Configurable, strategy-driven pipelines (expand_basic/expand_all pattern), b
ut implemented as typed transforms rather than arbitrary strategy hooks.
  - Multi-timeframe composites, shifted candles, correlation pairs: keep them te
acher-side/offline in pipeline and distill into student “soft labels.” Do not ru
n these heavy transforms in the hot path.
  - Pipeline and label pipelines separation: keep label generation separate and
manifest it for reproducibility.
- What to avoid in hot path:
  - Arbitrary Pandas operations; stick to Numpy + Nautilus indicators.
  - Expensive or variable-time transforms online (correlations, multi-timeframe
merges).
- Safety and parity:
  - Ensure transforms have identical math between batch and online (like we do t
oday).
  - Always record feature order and schema hash; actor validates at load.

**Contracts and Testing**

- Contracts
  - StudentFeatureSetContract:
    - DataRequirements == L1_ONLY
    - Hot path P99 features < 0.5 ms on reference hardware
    - schema_hash present; parity diff ≤ 1e-10
  - TeacherFeatureSetContract:
    - May use L2/L3 and complex transforms
    - Distillation target well-defined; links to student models
- Tests
  - Property tests for feature invariants (no NaNs, bounded normalized RSI, etc.
).
  - Parity test fixture against synthetic OHLCV.
  - Latency microbench that exercises the exact pipeline declared in manifest.
  - Registry contract tests for creation/promotion/deprecation.

**Minimal Implementation Plan**

- Feature manifests and registry
  - Add `ml/registry/feature_registry.py`:
    - `FeatureRole`, `FeatureManifest`, `LocalFeatureRegistry`.
  - Add `ml/docs/UNIFIED_FEATURE_REGISTRY.md` documenting schema and lifecycle (
parallel to model registry doc).
- Pipeline interface
  - Add `ml/features/pipeline.py`:
    - `FeatureTransform` base, catalog, and a thin runner that calls into curren
t FeatureEngineer stages.
    - Start by wrapping existing FeatureEngineer sections into discrete transfor
ms (returns/momentum, volatility, indicators, optional bundles).
  - Add a manifest generator:
    - Compute `feature_names`, `dtypes`, `schema_hash`, `pipeline_signature`.
- Training integration
  - Extend student training to:
    - Save `feature_set.manifest.json`, `scaler.npz`, `parity_report.json`.
    - Register feature set and link returned `feature_set_id` into the model’s `
ModelManifest`.
- Actor integration
  - Extend `MLSignalActorConfig`:
    - `feature_set_id`, `registry_path`, `use_registry_features`.
  - On initialization:
    - Load FeatureManifest; configure FeatureEngineer; assert schema; log `featu
re_set_id`.
- Telemetry
  - Add Prometheus metrics with `feature_set_id` label and simple drift hooks (o
ptional first version).

**Governance & Ops**

- Versioning and compatibility:
  - Semantic version for pipeline engine; store in manifest.
  - Backward compatibility rules: adding features increments minor, removing or
reordering increments major; actor refuses to serve mismatched major versions.
- Docs and discoverability:
  - Auto-generate `api_reference` and manifest examples for each shipped feature
 set.
- Security and stability:
  - No pickle in feature artifacts; only json/npz.
  - No network access in hot path; unit-test harden all transforms.

**Answering Your Question (“we’ll need a feature registry system and pipeline mu
ch like our model one?”)**

- Yes. Implement a Feature Registry and declarative feature pipeline that mirror
s the Model Registry’s rigor:
  - Automated creation from training, strict parity and latency validation, prom
otion gates, deprecation/scrapping, lineage tying models to feature sets, and ru
ntime schema/hash validation by the actor.

**Next Steps (Practical, Incremental)**

- Phase 1
  - Define `FeatureManifest` + `LocalFeatureRegistry`.
  - Add manifest generation to current FeatureEngineer; store schema/hash; expos
e `dump_manifest()`.
  - Update actor to optionally consume `feature_set_id` and manifest-based confi
guration.
- Phase 2
  - Extract FeatureEngineer stages into `FeatureTransform`s under `ml/features/p
ipeline.py` with a catalog.
  - Add registry-backed lifecycle commands and CLI helpers (create/promote/depre
cate).
  - Add parity and latency gates into registry promotion flow.
- Phase 3
  - Add drift monitors and A/B canary of feature sets (e.g., compare two feature
_set_ids under same model).
  - Introduce teacher-only pipeline transforms (multi-timeframe, corr pairs) and
 keep student hot path lean.
  - Expand docs and add contract tests.

-------------------

VERSION 3:

- Build a first-class Feature Registry and Pipeline with parity, latency, and li
fecycle gates.
- Integrate with training and MLSignalActor for manifest-driven, schema-validate
d online inference.
- Validate via Hypothesis-driven functional tests and contract tests.

**Modules & Subtasks**

- ml/registry/feature_registry.py
  - FeatureRole: enum for TEACHER, STUDENT, INFERENCE_SUPPORT.
  - FeatureManifest: type-safe manifest (identity, role, data_requirements, feat
ure_schema, schema_hash, pipeline_signature, constraints, parity/perf digests, l
ineage, metadata).
  - LocalFeatureRegistry:
    - register_feature_set(manifest, artifacts)
    - promote(feature_set_id, stage)
    - deprecate(feature_set_id, reason)
    - scrap(feature_set_id)
    - get_feature_set(feature_set_id)
    - resolve_by_schema_hash(schema_hash)
    - list_by_role(role), list_by_data_requirements(req), get_lineage(feature_se
t_id)
  - Validators:
    - FeatureContractValidator for Student/Teacher contracts.
    - Manifest integrity and schema hash determinism.

- ml/features/pipeline.py
  - FeatureTransform: base interface (name, requires, applies_to, fit, transform
_batch, transform_online, stateful).
  - TransformCatalog: registration and lookup.
  - PipelineSpec: declarative list of transforms + parameters.
  - PipelineRunner:
    - compile(pipeline_spec, capabilities)
    - run_batch(df) respecting parity rules (using IndicatorManager sequence whe
n needed)
    - run_online(current_bar, indicator_manager, buffer) zero-allocation, pre-al
located order
    - compute_feature_names(), dtypes; compute pipeline_signature.

- ml/features/engineering.py (augment)
  - compute_schema_hash(feature_names, dtypes) stable hashing.
  - dump_feature_manifest(pipeline_spec, role, data_requirements, constraints, r
eports) -> FeatureManifest.
  - expose get_feature_names() and ensure deterministic ordering vs PipelineRunn
er.

- ml/actors/signal.py (augment)
  - MLSignalActorConfig: add feature_set_id, registry_path, use_registry_feature
s.
  - At init:
    - If feature_set_id provided: load FeatureManifest from LocalFeatureRegistry
.
    - Validate constraints (L1_ONLY for Student, latency budgets if known).
    - Configure FeatureEngineer/PipelineRunner with manifest’s schema order, ass
ert exact equality and schema_hash; fail fast if mismatch.
  - Emit metrics with feature_set_id labels.

- ml/training/base.py (augment)
  - After batch feature build:
    - Fit scaler (train-split only), compute parity (FeatureParityValidator), la
tency report.
    - Create FeatureManifest (+ artifacts: scaler.npz, pipeline.json, parity_rep
ort.json).
    - Register in LocalFeatureRegistry; return feature_set_id.
  - Link feature_set_id into ModelManifest (model registry).

- ml/monitoring/ (optional)
  - Drift monitors (PSI/KS hooks) with feature_set_id labeling.

**Functional Tests (Hypothesis)**

- ml/tests/unit/registry/test_feature_manifest.py
  - test_manifest_schema_hash_determinism
    - Create multiple FeatureManifests with same feature_names/dtypes/pipeline_s
ignature; assert equal schema_hash.
    - Vary order; assert different schema_hash.
    - Hypothesis: generate random feature name lists and dtypes.

    `@given(names=st.lists(st.text(min_size=1, max_size=20), min_size=1, max_siz
e=64, unique=True))`
  - test_manifest_contract_student_l1_only
    - Manifest with role=STUDENT, data_requirements=L1_ONLY; validator passes.
    - Any transform requiring L2 yields validator failure.
  - test_manifest_roundtrip_json
    - Serialize/deserialize; schema_hash stable; datetime fields preserved.

- ml/tests/unit/registry/test_local_feature_registry.py
  - test_register_promote_deprecate_scrap
    - Register returns id; promote to ‘prod’; deprecate and scrap; errors on loa
d after scrap.
  - test_resolve_by_schema_hash
    - Register two manifests with same schema_hash; resolve returns latest prefe
rred or list; assert deterministic policy.
  - test_lineage_queries
    - Parent (teacher) and child (student) linkage accessible.

- ml/tests/unit/features/test_pipeline_core.py
  - test_pipeline_compilation_gating
    - PipelineSpec with L2-required transform compiled under Student(L1_ONLY) ra
ises error.
  - test_compute_feature_names_ordering_stability
    - Given pipeline spec and seed parameters, compute_feature_names returns sta
ble, deterministic list; hashing stable.
  - test_online_zero_allocation
    - Run transform_online repeatedly; memory allocations bounded (if feasible v
ia tracemalloc) and dtype float32.
  - test_pipeline_signature_changes_on_params
    - Alter transform parameter; signature changes; schema_hash changes.

- ml/tests/unit/features/test_feature_engineering_parity.py
  - test_parity_batch_online
    - Generate synthetic OHLCV (Hypothesis strategy) with plausible values.
    - Warm indicators, compute batch features (fit_scaler=False), online feature
s per-bar; assert max diff ≤ tolerance across all features.
  - test_rsi_bounds_and_normalization
    - Assert RSI normalized in [-1, 1]; raw back-conversion in [0, 100].
  - test_macd_signal_handling
    - With approximate_macd_signal True: signal/diff computed; False: zeros; con
sistent batch/online.

- ml/tests/integration/test_feature_registry_integration.py
  - test_end_to_end_register_and_use_in_actor
    - Build pipeline spec; compute features batch; parity validate; latency repo
rt stub.
    - Register FeatureManifest; create a dummy ModelManifest referencing feature
_set_id.
    - Initialize MLSignalActor with feature_set_id; assert schema validated; gen
erate signals on synthetic bars.

- ml/tests/perf/test_pipeline_latency.py
  - test_hot_path_latency_budget
    - Run N iterations online using pre-allocated buffers; assert p99 latency <
configured budget (allow configurable skip in CI).
  - test_batch_parity_latency_regression
    - Compare against baseline numbers (optional thresholds).

- ml/tests/property/test_feature_contracts.py
  - test_student_contract_l1_only_property
    - Hypothesis: random PipelineSpec with transforms flagged L1/L2; assert cont
racts reject any L2 under Student.

**Test Code Sketches**

- Manifest hash determinism

`from hypothesis import given, strategies as st
from ml.registry.feature_registry import FeatureManifest

@given(st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=32, uniqu
e=True))
def test_manifest_schema_hash_determinism(names):
    dtypes = ["float32"] * len(names)
    m1 = FeatureManifest(feature_names=names, feature_dtypes=dtypes, pipeline_si
gnature="sigA", ...)
    m2 = FeatureManifest(feature_names=list(names), feature_dtypes=list(dtypes),
 pipeline_signature="sigA", ...)
    assert m1.schema_hash == m2.schema_hash
    # Reorder -> different hash
    if len(names) > 1:
        swapped = names[::-1]
        m3 = FeatureManifest(feature_names=swapped, feature_dtypes=dtypes, pipel
ine_signature="sigA", ...)
        assert m1.schema_hash != m3.schema_hash
`

- Pipeline gating

`def test_pipeline_compilation_gating(student_pipeline_spec, transform_catalog):
    # Assume spec includes an L2-only transform
    with pytest.raises(ValueError):
        PipelineRunner.compile(student_pipeline_spec, capabilities={"L1_ONLY"})
`

- Parity property

`from hypothesis import given
import numpy as np
import pandas as pd

def ohlcv_frames():
    n = st.integers(min_value=80, max_value=200)
    def arr():
        return st.lists(st.floats(min_value=10, max_value=1000, allow_nan=False,
 allow_infinity=False), min_size=1, max_size=200)
    # Build consistent OHLCV arrays with high >= max(open, close), low <= min(op
en, close)
    ...

@given(df=ohlcv_frames())
def test_parity_batch_online(df):
    cfg = FeatureConfig()
    fe = FeatureEngineer(cfg)
    batch_df, _ = fe.calculate_features_batch(df, fit_scaler=False)
    mgr = IndicatorManager(cfg)
    # Warm-up first K rows
    # Compute online per bar; copy buffers
    # Compare arrays over range [start:end] with tolerance
`

- Actor integration

`def test_actor_uses_feature_set_manifest(local_feature_registry, feature_manife
st, synthetic_bars):
    feature_set_id = local_feature_registry.register_feature_set(feature_manifes
t, artifacts)
    actor_cfg = MLSignalActorConfig(feature_set_id=feature_set_id, registry_path
=str(local_registry_path), use_registry_features=True, ...)
    actor = MLSignalActor(actor_cfg)
    # Feed bars and assert predictions and metrics contain feature_set_id
    assert actor._feature_engineer.n_features == len(feature_manifest.feature_na
mes)
`

- Registry lifecycle

`def test_register_promote_deprecate_scrap(tmp_path, manifest):
    reg = LocalFeatureRegistry(tmp_path)
    fsid = reg.register_feature_set(manifest, artifacts)
    reg.promote(fsid, "prod")
    m = reg.get_feature_set(fsid)
    assert m.stage == "prod"
    reg.deprecate(fsid, "replaced")
    reg.scrap(fsid)
    with pytest.raises(KeyError):
        reg.get_feature_set(fsid)
`

**Hypothesis Strategies (shared fixtures)**

- OHLCV generator (monotonic timestamps, sane OHLCV relationships, nonnegative v
olumes).
- PipelineSpec generator (bounded transforms; forbid L2 transforms when testing
Student contracts).
- Feature names generator (valid snake_case strings with max length; uniqueness)
.

**Gates & CI**

- Lint/type: `make ruff`; `uv run --active --no-sync mypy ml --strict`.
- Tests: `make pytest`; coverage ≥ 90% for ml/.
- Performance tests: Mark with `@pytest.mark.perf` and allow skip or relaxed thr
esholds in CI; strict in perf job.
- Contracts: Failing StudentFeatureSetContract should fail CI.

**Migration Plan**

- Phase 1: Implement FeatureManifest and LocalFeatureRegistry with minimal API,
add manifest dump to FeatureEngineer, and actor validation logic.
  - Tests: manifest determinism, registry lifecycle, actor load/validation.
- Phase 2: Introduce PipelineSpec and PipelineRunner (wrap existing FeatureEngin
eer stages), capability gating, and parity tests driven by pipeline.
  - Tests: gating, parity, signature/hash, online zero-allocation.
- Phase 3: Training integration (register feature sets from trainers) and teleme
try with feature_set_id.
  - Tests: end-to-end registry+actor integration, telemetry labels.
- Phase 4: Optional teacher-only transforms (multi-timeframe, corr-pairs) and dr
ift monitors.
  - Tests: contract enforcement, drift hooks.

Would you like me to scaffold the FeatureManifest/LocalFeatureRegistry and add t
he first wave of tests under ml/tests now, then iterate on PipelineSpec/Runner i
n Phase 2?
