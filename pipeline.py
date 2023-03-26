# for encoding categorical variables
from feature_engine.encoding import OneHotEncoder, RareLabelEncoder

# for imputation
from feature_engine.imputation import AddMissingIndicator, MeanMedianImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# pipeline
from sklearn.pipeline import Pipeline

# feature scaling
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from heart_model.config.core import config

# set up the pipeline
heart_pipe = Pipeline(
    [
        # ===== IMPUTATION =====
        # add missing indicator to numerical variables
        ("missing_indicator", AddMissingIndicator(variables=config.model_config.numerical_vars_with_na)),
        # impute numerical variables with the median
        (
            "median_imputation",
            MeanMedianImputer(imputation_method="median", variables=config.model_config.numerical_vars_with_na),
        ),
        # == CATEGORICAL ENCODING ======
        # remove categories present in less than 5% of the observations (0.05)
        # group them in one category called 'Rare'
        (
            "rare_label_encoder",
            RareLabelEncoder(tol=0.05, n_categories=1, variables=config.model_config.categorical_vars),
        ),
        # encode categorical variables using one hot encoding into k-1 variables
        ("categorical_encoder", OneHotEncoder(drop_last=True, variables=config.model_config.categorical_vars)),
        # scale
        ("scaler", StandardScaler()),
    ]
)

logit_step = [
    (
        "Logit",
        LogisticRegression(
            C=config.model_config.alpha, solver="liblinear", random_state=config.model_config.random_state
        ),
    ),
]


k_neighbors_step = [
    (
        "KNeighbors",
        KNeighborsClassifier(n_neighbors=config.model_config.n_neighbors, p=config.model_config.p),
    ),
]

svc_step = [
    (
        "SVC",
        SVC(C=config.model_config.C, random_state=config.model_config.random_state),
    ),
]

tree_step = [
    (
        "DecisionTree",
        DecisionTreeClassifier(random_state=config.model_config.random_state),
    ),
]
