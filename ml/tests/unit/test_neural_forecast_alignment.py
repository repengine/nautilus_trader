import sys
import types
import sys
import types

import pandas as pd
import polars as pl
import pytest
import numpy as np


class FeatureConfig:
    pass


class _DummyScaler:
    def transform(self, data):
        return data


class _DummyFeatureEngineer:
    def __init__(self, config):
        self.config = config

    def calculate_features_batch(self, data, fit_scaler=True):
        return pl.DataFrame(), _DummyScaler()

    def get_feature_names(self):
        return []


# Patch feature_engineering module before importing trainer
feature_engineering = types.ModuleType("ml.features.feature_engineering")
feature_engineering.FeatureConfig = FeatureConfig
feature_engineering.FeatureEngineerV2 = _DummyFeatureEngineer
sys.modules["ml.features.feature_engineering"] = feature_engineering

class _DummyCuda:
    @staticmethod
    def is_available():
        return False

sys.modules["torch"] = types.SimpleNamespace(cuda=_DummyCuda())

sk_metrics = types.SimpleNamespace(
    mean_absolute_error=lambda y_true, y_pred: float(np.mean(np.abs(np.array(y_true) - np.array(y_pred)))),
    mean_absolute_percentage_error=lambda y_true, y_pred: float(
        np.mean(np.abs((np.array(y_true) - np.array(y_pred)) / np.array(y_true)))
    ),
    mean_squared_error=lambda y_true, y_pred: float(
        np.mean((np.array(y_true) - np.array(y_pred)) ** 2)
    ),
)
sys.modules["sklearn"] = types.SimpleNamespace(metrics=sk_metrics)
sys.modules["sklearn.metrics"] = sk_metrics

config_pkg = types.ModuleType("ml.config")
config_pkg.__path__ = []
sys.modules["ml.config"] = config_pkg

settings_module = types.ModuleType("ml.config.settings")
class Settings:
    pass
settings_module.Settings = Settings
sys.modules["ml.config.settings"] = settings_module

data_pkg = types.ModuleType("ml.data")
data_pkg.__path__ = []
sys.modules["ml.data"] = data_pkg

unified_loader_module = types.ModuleType("ml.data.unified_loader")
class UnifiedNautilusDataLoader:
    pass
unified_loader_module.UnifiedNautilusDataLoader = UnifiedNautilusDataLoader
sys.modules["ml.data.unified_loader"] = unified_loader_module

resource_pkg = types.ModuleType("ml.resource_management")
resource_pkg.__path__ = []
sys.modules["ml.resource_management"] = resource_pkg

trainer_mixin_module = types.ModuleType("ml.resource_management.trainer_mixin")
class ResourceManagedTrainerMixin:
    pass
trainer_mixin_module.ResourceManagedTrainerMixin = ResourceManagedTrainerMixin
sys.modules["ml.resource_management.trainer_mixin"] = trainer_mixin_module

utils_pkg = types.ModuleType("ml.utils")
utils_pkg.__path__ = []
sys.modules["ml.utils"] = utils_pkg

df_converter_module = types.ModuleType("ml.utils.dataframe_converter")
class DataFrameConverter:
    @staticmethod
    def polars_to_pandas(df):
        return pd.DataFrame({col: df[col].to_list() for col in df.columns})
df_converter_module.DataFrameConverter = DataFrameConverter
sys.modules["ml.utils.dataframe_converter"] = df_converter_module

mlflow_utils_module = types.ModuleType("ml.utils.mlflow_utils")
class MLflowManager:
    def __init__(self, *args, **kwargs):
        pass
    def set_experiment(self, *args, **kwargs):
        pass
    def start_run(self, *args, **kwargs):
        class Dummy:
            def __enter__(self_inner):
                return self_inner
            def __exit__(self_inner, exc_type, exc, tb):
                pass
            def log_params(self_inner, *a, **k):
                pass
            def log_metrics(self_inner, *a, **k):
                pass
            def log_model(self_inner, *a, **k):
                pass
            def log_artifact(self_inner, *a, **k):
                pass
            def log_dict_as_json(self_inner, *a, **k):
                pass
        return Dummy()
    def log_params(self, *a, **k):
        pass
    def log_metrics(self, *a, **k):
        pass
    def log_model(self, *a, **k):
        pass
    def log_artifact(self, *a, **k):
        pass
    def log_dict_as_json(self, *a, **k):
        pass
mlflow_utils_module.MLflowManager = MLflowManager
sys.modules["ml.utils.mlflow_utils"] = mlflow_utils_module

training_pkg = types.ModuleType("ml.training")
training_pkg.__path__ = []
sys.modules["ml.training"] = training_pkg

base_trainer_module = types.ModuleType("ml.training.base_trainer")
class BaseTrainer:
    def __init__(self, config, settings=None):
        self.config = config
        self.settings = settings or types.SimpleNamespace(
            mlflow=types.SimpleNamespace(experiment_name="test")
        )
base_trainer_module.BaseTrainer = BaseTrainer
sys.modules["ml.training.base_trainer"] = base_trainer_module

nf_models_stub = types.ModuleType("neuralforecast.models")
for name in [
    "NBEATS",
    "NHITS",
    "TFT",
    "Informer",
    "Autoformer",
    "PatchTST",
    "DLinear",
    "NLinear",
    "TimesNet",
    "TSMixer",
]:
    setattr(nf_models_stub, name, type(name, (), {}))
nf_stub = types.ModuleType("neuralforecast")
class _TempNF:
    pass
nf_stub.NeuralForecast = _TempNF
sys.modules["neuralforecast"] = nf_stub
sys.modules["neuralforecast.models"] = nf_models_stub

import importlib.util
from pathlib import Path

module_path = Path(__file__).resolve().parents[3] / "ml" / "training" / "neural_forecast.py"
spec = importlib.util.spec_from_file_location("ml.training.neural_forecast", module_path)
nf_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nf_module)


class _MockNeuralForecast:
    def __init__(self, models, freq):
        self.models = models
        self.freq = freq
        self.last_forecast_df = None

    def fit(self, df, val_size=None):
        self.df = df
        return self

    def predict(self, df=None, h=None):
        ds = df["ds"].iloc[-h:].reset_index(drop=True)
        preds = df["y"].iloc[-h:] + 1.0
        self.last_forecast_df = pd.DataFrame(
            {
                "unique_id": df["unique_id"].iloc[-h:].values,
                "ds": ds,
                "MockModel": preds.values,
            }
        )
        return self.last_forecast_df


nf_module.NEURALFORECAST_AVAILABLE = True
nf_module.NeuralForecast = _MockNeuralForecast

NeuralForecastTrainer = nf_module.NeuralForecastTrainer


def test_validation_forecast_alignment(monkeypatch):
    trainer = NeuralForecastTrainer(
        config={"model_type": "NLinear", "forecast_horizon": 2, "input_size": 2, "freq": "D"}
    )
    monkeypatch.setattr(trainer, "_create_model", lambda params: "model")

    train_pl = pl.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="D"),
            "close": [1.0, 2.0, 3.0],
        }
    )
    val_pl = pl.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-04", periods=2, freq="D"),
            "close": [4.0, 5.0],
        }
    )

    results = trainer._train_with_params(train_pl, val_pl, params={})
    forecast_df = results["model"].last_forecast_df

    expected_ds = pd.to_datetime(val_pl["timestamp"].to_list()).tolist()
    assert forecast_df["ds"].tolist() == expected_ds
    assert results["metrics"]["val_mae"] == pytest.approx(1.0)
