import typing as t

import pandas as pd

from heart_model import __version__ as _version
from heart_model.config.core import config
from heart_model.processing.data_manager import load_pipeline
from heart_model.processing.validation import validate_inputs

pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
_heart_pipe = load_pipeline(file_name=pipeline_file_name)


def make_prediction(
    *,
    input_data: t.Union[pd.DataFrame, dict],
) -> dict:
    """Make a prediction using a saved model pipeline."""

    data = pd.DataFrame(input_data)
    validated_data, errors = validate_inputs(input_data=data)

    results: t.Dict[str, t.Any] = {"preds": None, "probs": None, "version": _version, "errors": errors}

    preds = _heart_pipe.predict(X=validated_data[config.model_config.features])
    probs = _heart_pipe.predict_proba(X=validated_data[config.model_config.features])[:, 1]

    # Fill the results dict
    results["preds"] = [pred for pred in preds]
    results["probs"] = [prob for prob in probs]

    return results