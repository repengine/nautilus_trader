"""
Nautilus ML Prototypes
==========================

This module contains prototype classes and functions for extending
`NautilusTrader` with state-of-the-art machine learning models and
data preprocessing utilities.  The goal of these prototypes is to
demonstrate how advanced techniques outlined in the research report
on modern quantitative finance can be integrated into the Nautilus
framework.  While the implementations here are intentionally
lightweight and illustrative, they are structured to align with
NautilusTrader’s AI-first, event-driven architecture: models are
registered in a central registry, features are declared and
retrievable via a feature registry, and each model exposes a common
interface for training and inference.  These prototypes take
inspiration from the existing code base (e.g. ``ModelRegistry``,
``FeatureRegistry``, ``Signal``, and ``MLStrategy``) while remaining
self-contained so they can be iterated on and fleshed out within
your IDE.

The NautilusTrader README explains that the platform is designed to
"develop and deploy algorithmic trading strategies within a highly
performant and robust Python-native environment"【210236011427384†L31-L37】.  It
also notes that Nautilus is "AI-first" and provides a backtest
engine "fast enough to be used to train AI trading agents"【210236011427384†L35-L37】【210236011427384†L57-L61】.
These prototypes aim to capitalise on that design philosophy by
introducing models such as Transformers (Autoformer/Informer),
Deep LOB CNNs, Reinforcement Learning (RL) agents, Graph Neural
Networks (GNNs), State-Space Models (SSMs), N-BEATS, and
Generative models for synthetic data.  In addition, we provide
preprocessing utilities (e.g. fractional differencing) to help
ensure that the inputs to these models satisfy the stationarity
assumptions required for many statistical learning algorithms.

To use these prototypes, register each model with the ``ModelRegistry``
and register any derived features with the ``FeatureRegistry``.  The
``MLStrategy`` base class illustrates how a trading strategy can
consume models and features to generate signals.  All code here
should be considered experimental – you are encouraged to extend the
classes, implement the ``fit``/``predict`` methods using your
preferred deep learning frameworks (PyTorch, TensorFlow, JAX, etc.),
and integrate more advanced mechanisms (e.g. cost models, regime
switching, meta-labeling) as needed.
"""

from __future__ import annotations

import abc
import dataclasses
from collections import defaultdict
from collections.abc import Callable
from collections.abc import Iterable
from dataclasses import field
from typing import Any

import numpy as np


class BaseModel(abc.ABC):
    """
    Abstract base class for all ML models.

    Subclasses must implement :meth:`fit` and :meth:`predict`.  The
    ``fit`` method trains the model on historical data, while
    ``predict`` produces model outputs (e.g. probabilities, returns,
    signals) given new features.  Additional methods such as
    ``update`` or ``retrain`` can be added for online/continual
    learning.

    """

    @abc.abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the model on input matrix ``X`` and target vector ``y``.

        Parameters
        ----------
        X : np.ndarray
            The feature matrix.  Shape = (n_samples, n_features).
        y : np.ndarray
            The target values.  Shape = (n_samples,) or (n_samples, n_targets).

        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return model predictions for the input matrix ``X``.

        Parameters
        ----------
        X : np.ndarray
            The feature matrix.  Shape = (n_samples, n_features).

        Returns
        -------
        np.ndarray
            The model’s predictions.

        """
        raise NotImplementedError

    def update(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Optional hook for incremental/online updates.

        Default implementation simply calls :meth:`fit`.  Override
        this method to perform incremental fitting without retraining
        from scratch.

        """
        self.fit(X, y)


class ModelRegistry:
    """
    Singleton registry for model classes.

    Use this registry to map string identifiers (e.g. ``"autoformer"``)
    to model classes.  Models are registered via the
    :func:`register` decorator or the :meth:`register_model` method.

    """

    _models: dict[str, Callable[..., BaseModel]] = {}

    @classmethod
    def register_model(cls, name: str, model_cls: Callable[..., BaseModel]) -> None:
        """
        Register a model class under a given name.

        Parameters
        ----------
        name : str
            The unique identifier for the model.
        model_cls : Callable[..., BaseModel]
            The class implementing the model.

        """
        if name in cls._models:
            raise KeyError(f"Model {name!r} already registered.")
        cls._models[name] = model_cls

    @classmethod
    def create(cls, name: str, *args: Any, **kwargs: Any) -> BaseModel:
        """
        Instantiate a registered model.

        Parameters
        ----------
        name : str
            The identifier of the model.
        args, kwargs : Any
            Arguments passed to the model’s constructor.

        Returns
        -------
        BaseModel
            An instance of the requested model.

        """
        model_cls = cls._models.get(name)
        if model_cls is None:
            raise KeyError(f"Model {name!r} is not registered.")
        return model_cls(*args, **kwargs)

    @classmethod
    def list_models(cls) -> list[str]:
        """
        Return a list of all registered model names.
        """
        return list(cls._models.keys())


def register_model(name: str) -> Callable[[Callable[..., BaseModel]], Callable[..., BaseModel]]:
    """
    Decorator for registering a model class.

    Example
    -------
    >>> @register_model("my_model")
    ... class MyModel(BaseModel):
    ...     ...

    """

    def decorator(cls: Callable[..., BaseModel]) -> Callable[..., BaseModel]:
        ModelRegistry.register_model(name, cls)
        return cls

    return decorator


class FeatureRegistry:
    """
    Registry for feature engineering functions.

    Functions registered here can be called by name to transform raw market data into
    ML-ready features.  This encourages re-use and centralisation of feature
    definitions.

    """

    _features: dict[str, Callable[..., np.ndarray]] = {}

    @classmethod
    def register_feature(cls, name: str, func: Callable[..., np.ndarray]) -> None:
        if name in cls._features:
            raise KeyError(f"Feature {name!r} already registered.")
        cls._features[name] = func

    @classmethod
    def get(cls, name: str) -> Callable[..., np.ndarray]:
        func = cls._features.get(name)
        if func is None:
            raise KeyError(f"Feature {name!r} not found in registry.")
        return func

    @classmethod
    def list_features(cls) -> list[str]:
        return list(cls._features.keys())


@dataclasses.dataclass
class Signal:
    """
    Represents an actionable signal produced by a model.

    A signal might encapsulate directional information (long/short), confidence scores,
    or expected returns.  Additional fields such as stop-loss/take-profit levels,
    execution instructions, or probabilities can be added as required.

    """

    timestamp: int
    instrument_id: str
    value: float
    probability: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class MLStrategy(abc.ABC):
    """
    Base class for ML-driven trading strategies.

    A strategy consumes market data, produces features via the
    ``FeatureRegistry``, feeds them into one or more models from the
    ``ModelRegistry``, and emits trading signals.  Subclasses must
    implement the :meth:`on_bar` (or :meth:`on_tick`, etc.) callback,
    which is invoked by the Nautilus engine with each new data
    update.

    """

    def __init__(self, model_name: str, feature_names: list[str]):
        self.model = ModelRegistry.create(model_name)
        self.feature_names = feature_names

    def on_bar(self, bar_data: dict[str, Any]) -> list[Signal]:
        """
        Process a bar of market data and return a list of signals.

        Parameters
        ----------
        bar_data : Dict[str, Any]
            Raw bar data (e.g. OHLCV, order book snapshots).  The
            concrete format depends on your data pipeline.

        Returns
        -------
        List[Signal]
            A list of signals to be sent to the execution engine.

        """
        # Extract features using the registered feature functions.
        features = []
        for name in self.feature_names:
            feature_fn = FeatureRegistry.get(name)
            features.append(feature_fn(bar_data))
        X = np.column_stack(features)
        preds = self.model.predict(X)
        # Convert predictions to signals.  Here we assume ``preds`` is an
        # array of floats representing expected returns; positive
        # values produce long signals, negatives produce short.
        signals: list[Signal] = []
        ts = bar_data.get("timestamp")
        instrument = bar_data.get("instrument_id", "UNKNOWN")
        for value in np.atleast_1d(preds):
            prob = None
            if np.ndim(preds) == 2 and preds.shape[1] > 1:
                # Example: second column stores confidence or probability
                prob = float(value[1])  # type: ignore[index]
                value = float(value[0])  # type: ignore[assignment]
            signals.append(
                Signal(timestamp=ts, instrument_id=instrument, value=float(value), probability=prob)
            )
        return signals

    @abc.abstractmethod
    def train(self, historical_data: Iterable[dict[str, Any]]) -> None:
        """
        Train the underlying model using historical data.

        Implementations should assemble feature matrices and targets
        appropriate for the chosen model and call ``self.model.fit``.

        """
        raise NotImplementedError


###############################################################################
#                             Advanced Model Prototypes                       #
###############################################################################


@register_model("autoformer")
class AutoformerModel(BaseModel):
    """
    Prototype for an Autoformer time-series forecasting model.

    Autoformer is a transformer architecture designed for long sequences.  It uses a
    series decomposition mechanism to separate trend and seasonal components, then
    applies an encoder–decoder structure with autocorrelation based attention.  This
    prototype omits the full implementation but outlines the core interfaces needed for
    training and inference.  Use PyTorch or TensorFlow to implement the actual layers.

    """

    def __init__(
        self, input_dim: int = 1, hidden_dim: int = 64, n_layers: int = 2, **kwargs: Any
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        # Placeholder for the underlying neural network.  In a real
        # implementation this might be a torch.nn.Module.
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:  # type: ignore[override]
        # TODO: Build Autoformer architecture and train it on (X, y).
        # This might involve decomposing X into trend/seasonal parts and
        # minimising a forecasting loss (e.g. MAE).
        raise NotImplementedError("Autoformer fit not implemented")

    def predict(self, X: np.ndarray) -> np.ndarray:  # type: ignore[override]
        # TODO: Perform forward pass to obtain predictions.  For now we
        # return zeros of appropriate shape.
        return np.zeros((len(X), 1))


@register_model("informer")
class InformerModel(BaseModel):
    """
    Prototype for an Informer time-series forecasting model.

    Informer introduces ProbSparse self-attention and distillation mechanisms to
    efficiently handle very long input sequences.  It performs well for multi-horizon
    forecasting tasks.  Replace the stubbed methods with a proper implementation (e.g.
    using the official Informer repository) when integrating into your system.

    """

    def __init__(
        self, input_dim: int = 1, hidden_dim: int = 64, n_layers: int = 2, **kwargs: Any
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.model = None  # placeholder

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError("Informer fit not implemented")

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros((len(X), 1))


@register_model("cvml_orderbook")
class CVMLOrderBookModel(BaseModel):
    """
    Prototype for a Convolutional Cross-Variate Mixing Layer (CVML) model.

    This model operates on limit order book (LOB) data represented as a 3-D tensor of
    shape (time, levels, features).  A CVML applies convolutions across both the price
    levels and feature dimensions to capture spatial patterns, then mixes cross-variate
    information via depthwise convolutions.  DeepLOB and similar models use this
    structure.  Implement training/inference using a deep learning library; this
    prototype simply defines the interface.

    """

    def __init__(
        self, n_levels: int = 50, n_features: int = 4, hidden_dim: int = 64, **kwargs: Any
    ) -> None:
        self.n_levels = n_levels
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # TODO: Implement CNN layers to learn LOB dynamics.
        raise NotImplementedError("CVMLOrderBookModel fit not implemented")

    def predict(self, X: np.ndarray) -> np.ndarray:
        # TODO: Forward pass for LOB prediction.
        return np.zeros((len(X), 1))


@register_model("reinforcement_agent")
class ReinforcementLearningAgent(BaseModel):
    """
    Prototype for a deep reinforcement learning trading agent.

    RL agents learn to map observed market states to actions (e.g. buy/sell/hold) that
    maximise a reward function such as Sharpe ratio or risk-adjusted return.  In
    practice you might implement this using algorithms like Deep Q-Networks (DQN),
    Proximal Policy Optimisation (PPO), or Soft Actor–Critic (SAC).  This prototype
    captures the interface and provides a simple tabular fallback agent for
    demonstration.

    """

    def __init__(
        self, action_space: list[int] = [-1, 0, 1], state_dim: int = 10, **kwargs: Any
    ) -> None:
        self.action_space = action_space
        self.state_dim = state_dim
        # For the prototype we use a simple Q-table keyed by state
        # hashes.  Replace with a neural network for full DRL.
        self.q_table: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))

    def _state_to_key(self, state: np.ndarray) -> int:
        # Hash a continuous state into an integer key for tabular Q-learning
        return hash(state.tobytes())

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the agent offline using state/action/reward tuples.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix representing states; not used in this simple
            tabular example.
        y : np.ndarray
            Array of tuples (state, action, reward, next_state).

        """
        for state, action, reward, next_state in y:
            key = self._state_to_key(state)
            next_key = self._state_to_key(next_state)
            # Q-learning update: Q(s,a) = Q(s,a) + α [r + γ max_a' Q(s',a') – Q(s,a)]
            alpha = 0.1
            gamma = 0.99
            best_next = max(self.q_table[next_key].values() or [0.0])
            old_value = self.q_table[key][action]
            self.q_table[key][action] = old_value + alpha * (reward + gamma * best_next - old_value)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Select actions greedily from the Q-table for each state.
        """
        actions = []
        for state in X:
            key = self._state_to_key(state)
            # Choose the action with the highest Q-value; default to
            # 'hold' (0) if unseen.
            if self.q_table[key]:
                best_action = max(self.q_table[key], key=self.q_table[key].get)
            else:
                best_action = 0
            actions.append(best_action)
        return np.array(actions)


@register_model("graph_attention")
class GraphAttentionModel(BaseModel):
    """
    Prototype for a graph neural network (GNN) model with attention.

    This model captures cross-asset relationships by representing the market as a graph
    where nodes are instruments and edges encode dependencies (correlations, sector
    relationships, news co-mentions, etc.).  In a full implementation you would
    construct node embeddings via graph convolutions and apply temporal convolutions or
    transformers to capture dynamics.  Here we include only the interface.

    """

    def __init__(
        self, num_nodes: int = 100, input_dim: int = 10, hidden_dim: int = 32, **kwargs: Any
    ) -> None:
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError("GraphAttentionModel fit not implemented")

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros((len(X), 1))


@register_model("state_space")
class StateSpaceModel(BaseModel):
    """
    Prototype for a structured state-space model (SSM).

    SSMs (e.g. S4) model long-range dependencies through linear state dynamics and can
    handle extremely long sequences efficiently.  The architecture typically involves a
    convolution kernel parameterised via a structured matrix that implicitly defines the
    system’s impulse response.  This prototype leaves the implementation details to the
    user.

    """

    def __init__(self, input_dim: int = 1, hidden_dim: int = 64, **kwargs: Any) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError("StateSpaceModel fit not implemented")

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros((len(X), 1))


@register_model("nbeats")
class NBeatsModel(BaseModel):
    """
    Prototype for an N-BEATS model.

    N-BEATS is a deep residual network for time series forecasting composed of backward
    and forward fully connected blocks.  Each block outputs a forecast and a backcast
    component.  Replace the methods here with a concrete implementation to achieve
    state-of- the-art performance on univariate forecasting tasks.

    """

    def __init__(
        self, input_dim: int = 1, hidden_dim: int = 128, n_stacks: int = 3, **kwargs: Any
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_stacks = n_stacks
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError("NBeatsModel fit not implemented")

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros((len(X), 1))


@register_model("generative")
class GenerativeModel(BaseModel):
    """
    Prototype for a generative model for synthetic data.

    Generative models such as GANs or diffusion models can be used both to augment
    training data and to perform scenario analysis. This prototype provides a hook for
    training a generator on historical time series and sampling new synthetic sequences.
    For diffusion-based models consider the TimeDiffusion framework, which has shown
    promising results on financial data.

    """

    def __init__(self, input_dim: int = 1, latent_dim: int = 32, **kwargs: Any) -> None:
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.generator = None  # Placeholder for the generative network
        self.discriminator = None  # For GANs

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError("GenerativeModel fit not implemented")

    def predict(self, X: np.ndarray) -> np.ndarray:
        # For generative models, ``predict`` might return synthetic
        # sequences sampled from the learned distribution.  Here we
        # return zeros for demonstration.
        return np.zeros_like(X)


###############################################################################
#                        Data Preprocessing & Stationarity                    #
###############################################################################


def fractional_weights(d: float, size: int) -> np.ndarray:
    """
    Compute weights for fractional differencing.

    The weights decay according to the binomial series with exponent
    ``d`` and are used to apply fractional differencing to a time
    series.  See López de Prado (2018) for details.

    Parameters
    ----------
    d : float
        Differencing order.  Must be positive and typically less than 1.
    size : int
        Number of weights to compute.

    Returns
    -------
    np.ndarray
        The array of fractional differencing weights.

    """
    w = [1.0]
    for k in range(1, size):
        w_k = -w[-1] * (d - k + 1) / k
        w.append(w_k)
    return np.array(w[::-1])  # reverse for convolution


def fractional_difference(series: np.ndarray, d: float, threshold: float = 1e-3) -> np.ndarray:
    """
    Apply fractional differencing to a 1-D time series.

    Fractional differencing helps make a non-stationary series more
    stationary while retaining long-range memory【210236011427384†L71-L80】.  The
    algorithm convolves the series with weights computed via
    :func:`fractional_weights`.  Coefficients with absolute value
    smaller than ``threshold`` are dropped to improve efficiency.

    Parameters
    ----------
    series : np.ndarray
        Input time series of shape (n_samples,).
    d : float
        Differencing order.
    threshold : float, optional
        Minimum absolute weight magnitude to retain.

    Returns
    -------
    np.ndarray
        Fractionally differenced series of shape (n_samples,).

    """
    series = np.asarray(series, dtype=float)
    weights = fractional_weights(d, len(series))
    # Drop small weights
    idx = np.where(np.abs(weights) > threshold)[0]
    weights = weights[idx]
    out = np.zeros_like(series)
    for i in range(len(weights), len(series)):
        out[i] = np.dot(weights, series[i - len(weights) + 1 : i + 1])
    return out


def difference_series(series: np.ndarray, order: int = 1) -> np.ndarray:
    """
    Standard differencing for stationarity.

    A fallback when fractional differencing is not desired.  Computes
    ``np.diff`` repeatedly ``order`` times.

    """
    out = series.copy()
    for _ in range(order):
        out = np.diff(out, prepend=out[0])
    return out


def make_stationary(series: np.ndarray, method: str = "fractional", **kwargs: Any) -> np.ndarray:
    """
    Transform a time series to a (more) stationary version.

    Parameters
    ----------
    series : np.ndarray
        Input time series.
    method : str, optional
        Either ``"fractional"`` for fractional differencing, or
        ``"standard"`` for integer differencing.
    **kwargs : Any
        Additional parameters passed to the underlying differencing
        function (e.g. ``d`` for fractional differencing or ``order``
        for standard differencing).

    Returns
    -------
    np.ndarray
        Differenced series.

    """
    if method == "fractional":
        d = kwargs.get("d", 0.5)
        return fractional_difference(series, d)
    elif method == "standard":
        order = kwargs.get("order", 1)
        return difference_series(series, order=order)
    else:
        raise ValueError(f"Unknown stationarity method {method!r}")


def purged_walk_forward_cv(
    n_samples: int, n_splits: int = 5, test_size: float = 0.2, embargo_size: int = 0
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Generate purged walk-forward cross-validation splits.

    Purged CV partitions the data into sequential training/testing
    splits and removes overlapping samples to mitigate lookahead
    bias.  An optional embargo period excludes samples immediately
    following each training block.  Use this generator to iterate
    through train/test indices when evaluating time-series models.

    Parameters
    ----------
    n_samples : int
        Total number of observations.
    n_splits : int, optional
        Number of train/test splits.
    test_size : float, optional
        Fraction of the data to allocate to the test set in each split.
    embargo_size : int, optional
        Number of observations to embargo after each training fold.

    Returns
    -------
    List[Tuple[np.ndarray, np.ndarray]]
        List of (train_indices, test_indices) tuples.

    """
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    split_size = int(n_samples * test_size)
    for i in range(n_splits):
        test_start = i * split_size
        test_end = test_start + split_size
        train_indices = np.arange(0, test_start)
        # Apply embargo: drop the last ``embargo_size`` samples from the
        # training set if they overlap with the test window.
        if embargo_size > 0:
            train_indices = (
                train_indices[:-embargo_size]
                if len(train_indices) > embargo_size
                else np.array([], dtype=int)
            )
        test_indices = np.arange(test_start, min(test_end, n_samples))
        splits.append((train_indices, test_indices))
        if test_end >= n_samples:
            break
    return splits


def create_lagged_features(series: np.ndarray, lags: int = 5) -> np.ndarray:
    """
    Create a matrix of lagged features from a 1-D time series.

    For example, ``lags=3`` transforms ``[x0, x1, x2, x3, x4]`` into
    ``[[x0, x1, x2], [x1, x2, x3], [x2, x3, x4]]``.  Lagged features
    are a simple yet effective way to convert sequential data into a
    tabular format suitable for models that expect independent
    samples.

    """
    series = np.asarray(series)
    n = len(series)
    if n < lags:
        raise ValueError("Series length must be greater than number of lags")
    lagged = np.column_stack([series[i : n - lags + i] for i in range(lags)])
    return lagged


def standardise_features(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardise feature matrix to zero mean and unit variance.

    Returns the standardised matrix along with the estimated means and standard
    deviations for later use (e.g. on live data).

    """
    means = X.mean(axis=0)
    stds = X.std(axis=0) + 1e-8
    X_std = (X - means) / stds
    return X_std, means, stds


def normalise_series(series: np.ndarray) -> tuple[np.ndarray, float, float]:
    """
    Normalise a 1-D series to zero mean and unit variance.
    """
    mean = float(np.mean(series))
    std = float(np.std(series) + 1e-8)
    return (series - mean) / std, mean, std


###############################################################################
# Example Feature Registration                                                 #
###############################################################################


def mid_price_feature(bar_data: dict[str, Any]) -> np.ndarray:
    """
    Example feature: compute the mid-price from bid/ask.

    Assumes ``bar_data`` contains keys ``"bid"`` and ``"ask"``
    representing the best bid and best ask prices.  Returns a 1-D
    array since the ``FeatureRegistry`` expects functions to return
    arrays (column vectors) that can be stacked.
    """
    bid = float(bar_data.get("bid", 0.0))
    ask = float(bar_data.get("ask", 0.0))
    mid = (bid + ask) / 2
    return np.array([mid])


def spread_feature(bar_data: dict[str, Any]) -> np.ndarray:
    """Example feature: compute the bid–ask spread."""
    bid = float(bar_data.get("bid", 0.0))
    ask = float(bar_data.get("ask", 0.0))
    spread = ask - bid
    return np.array([spread])


# Register the example features
FeatureRegistry.register_feature("mid_price", mid_price_feature)
FeatureRegistry.register_feature("spread", spread_feature)


###############################################################################
#                               Module Exports                                #
###############################################################################

__all__ = [
    "BaseModel",
    "ModelRegistry",
    "FeatureRegistry",
    "Signal",
    "MLStrategy",
    # Model classes
    "AutoformerModel",
    "InformerModel",
    "CVMLOrderBookModel",
    "ReinforcementLearningAgent",
    "GraphAttentionModel",
    "StateSpaceModel",
    "NBeatsModel",
    "GenerativeModel",
    # Preprocessing functions
    "fractional_weights",
    "fractional_difference",
    "difference_series",
    "make_stationary",
    "purged_walk_forward_cv",
    "create_lagged_features",
    "standardise_features",
    "normalise_series",
]
