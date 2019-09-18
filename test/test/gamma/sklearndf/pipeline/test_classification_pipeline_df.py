import pandas as pd

# noinspection PyPackageRequirements
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

from gamma.sklearndf.classification import RandomForestClassifierDF
from gamma.sklearndf.pipeline import ClassifierPipelineDF
from test.gamma.sklearndf.pipeline import make_simple_transformer


def test_classification_pipeline_df(
    iris_features: pd.DataFrame, iris_target_sr: pd.DataFrame
) -> None:

    cls_p_df = ClassifierPipelineDF(
        classifier=RandomForestClassifierDF(),
        preprocessing=make_simple_transformer(
            impute_median_columns=iris_features.select_dtypes(
                include=pd.np.number
            ).columns,
            one_hot_encode_columns=iris_features.select_dtypes(include=object).columns,
        ),
    )

    cls_p_df.fit(X=iris_features, y=iris_target_sr)
    cls_p_df.predict(X=iris_features)

    # test-type check within constructor:
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        ClassifierPipelineDF(
            classifier=RandomForestClassifier(), preprocessing=OneHotEncoder()
        )
