import logging
import sys
from pathlib import Path

from config.core import LOG_DIR, config
from pipeline import heart_pipe, k_neighbors_step, logit_step, svc_step, tree_step
from processing.data_manager import load_dataset, save_pipeline
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from heart_model import __version__ as _version


def run_training(model_type: int) -> None:
    if model_type not in range(0, 3):
        print("argument must be in range [0, 3]")
        exit()
    """Train the model."""
    # Update logs
    log_path = Path(f"{LOG_DIR}/log_{_version}.log")
    if Path.exists(log_path):
        log_path.unlink()
    logging.basicConfig(filename=log_path, level=logging.DEBUG)

    # read training data
    data = load_dataset(file_name=config.app_config.training_data_file)

    data.rename(columns=config.model_config.variables_to_rename, inplace=True)
    data.drop(labels=config.model_config.variables_to_drop, axis=1, inplace=True)

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.model_config.features],  # predictors
        data[config.model_config.target].map({"yes": 1, "No": 0}),
        test_size=config.model_config.test_size,
        # we are setting the random seed here
        # for reproducibility
        random_state=config.model_config.random_state,
    )

    steps = [logit_step, k_neighbors_step, svc_step, tree_step]
    # fit model
    heart_pipe.steps.append(steps[model_type])
    heart_pipe.fit(X_train, y_train)

    # make predictions for train set
    class_ = heart_pipe.predict(X_train)
    pred = heart_pipe.predict_proba(X_train)[:, 1]

    # determine train accuracy and roc-auc
    train_accuracy = accuracy_score(y_train, class_)
    train_roc_auc = roc_auc_score(y_train, pred)

    print(f"train accuracy: {train_accuracy}")
    print(f"train roc-auc: {train_roc_auc}")
    print()

    logging.info(f"train accuracy: {train_accuracy}")
    logging.info(f"train roc-auc: {train_roc_auc}")

    # make predictions for test set
    class_ = heart_pipe.predict(X_test)
    pred = heart_pipe.predict_proba(X_test)[:, 1]

    # determine test accuracy and roc-auc
    test_accuracy = accuracy_score(y_test, class_)
    test_roc_auc = roc_auc_score(y_test, pred)

    print(f"test accuracy: {test_accuracy}")
    print(f"test roc-auc: {test_roc_auc}")
    print()

    logging.info(f"test accuracy: {test_accuracy}")
    logging.info(f"test roc-auc: {test_roc_auc}")

    # persist trained model
    save_pipeline(pipeline_to_persist=heart_pipe)


if __name__ == "__main__":
    run_training(int(sys.argv[1]))
