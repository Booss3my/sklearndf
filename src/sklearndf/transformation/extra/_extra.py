"""
Core implementation of :mod:`sklearndf.transformation.extra`
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import (
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
    r_regression,
)

from pytools.api import AllTracker

from ...wrapper import MissingEstimator
from .wrapper import ARFSWrapperDF as _ARFSWrapperDF
from .wrapper import BorutaPyWrapperDF as _BorutaPyWrapperDF

log = logging.getLogger(__name__)

__all__ = ["BoostAGrootaDF", "BorutaDF", "GrootCVDF", "LeshyDF", "MrmrDF"]

try:
    # import boruta classes only if installed
    from boruta import BorutaPy

    # Apply a hack to address boruta's incompatibility with numpy >= 1.24:
    # boruta uses np.float_ which is deprecated in numpy >= 1.20 and removed in 1.24.
    #
    # We check these types are already defined in numpy, and if not, we define them
    # as aliases to the corresponding new types with a trailing underscore.

    for __attr in ["bool", "int", "float"]:
        if not hasattr(np, __attr):
            setattr(np, __attr, getattr(np, f"{__attr}_"))
    del __attr

except ImportError:

    class BorutaPy(  # type: ignore
        MissingEstimator,
        TransformerMixin,  # type: ignore
    ):
        """Mock-up for missing estimator."""


try:
    # import boruta classes only if installed
    from arfs.feature_selection.allrelevant import BoostAGroota, GrootCV, Leshy

except ImportError:

    class BoostAGroota(  # type: ignore
        MissingEstimator,
        TransformerMixin,  # type: ignore
    ):
        """Mock-up for missing estimator."""

    class GrootCV(  # type: ignore
        MissingEstimator,
        TransformerMixin,  # type: ignore
    ):
        """Mock-up for missing estimator."""

    class Leshy(  # type: ignore
        MissingEstimator,
        TransformerMixin,  # type: ignore
    ):
        """Mock-up for missing estimator."""


#
# Ensure all symbols introduced below are included in __all__
#

__tracker = AllTracker(globals())


#
# Class definitions
#


class BorutaDF(_BorutaPyWrapperDF, native=BorutaPy):
    """
    DF version of :class:`~boruta.BorutaPy`.
    """


class LeshyDF(_ARFSWrapperDF[Leshy], native=Leshy):
    """
    DF version of :class:`~arfs.feature_selection.allrelevant.Leshy`.
    """


class BoostAGrootaDF(_ARFSWrapperDF[BoostAGroota], native=BoostAGroota):
    """
    DF version of :class:`~arfs.feature_selection.allrelevant.BoostAGroota`.
    """

    @property
    def estimator(self) -> BaseEstimator:
        """
        Alias for the native estimator's :attr:`.est` attribute, to conform with
        the :class:`~sklearn.base.MetaEstimatorMixin` interface.

        :return: the value of the native estimator's :attr:`.est` attribute
        """
        return self.native_estimator.est

    @estimator.setter
    def estimator(self, est: BaseEstimator) -> None:
        """
        Alias for the native estimator's :attr:`.est` attribute, to conform with
        the :class:`~sklearn.base.MetaEstimatorMixin` interface.

        :param est: the new value for the native estimator's :attr:`.est` attribute
        """
        self.native_estimator.est = est

    @estimator.deleter
    def estimator(self) -> None:
        """
        Alias for the native estimator's :attr:`.est` attribute, to conform with
        the :class:`~sklearn.base.MetaEstimatorMixin` interface.
        """
        del self.native_estimator.est


class GrootCVDF(_ARFSWrapperDF[GrootCV], native=GrootCV):
    """
    DF version of :class:`~arfs.feature_selection.allrelevant.GrootCV`.
    """


class MrmrDF(
    BaseEstimator,  # type: ignore
    TransformerMixin,  # type: ignore
):
    """
    Implementation of the mRMR (minimum Redundancy Maximum
    Relevance) feature selector.

    Parameters:
    ----------
    relevance : default="mi"
        The method to compute the relevance of each feature.
        Options are "mi" for mutual information and "fs" for f statistic.

    redundancy : default="mi"
        The method to compute the redundancy of each feature.
        Options are "mi" for mutual information and "pc" for Peason correllation.

    scheme : default="difference"
        The scheme to combine relevance and redundancy.
        Options are "difference" and "quotient".

    target_type : default="discrete"
        The type of the target variable.
        Options are "discrete" and "continuous".

    n_features_out : default=15
        The number of features to select.
    """

    def __init__(
        self,
        relevance: str = "mi",
        redundancy: str = "mi",
        scheme: str = "difference",
        target_type: str = "discrete",
        n_features_out: int = 15,
    ):
        self.relevance = relevance
        self.redundancy = redundancy
        self.scheme = scheme
        self.target_type = target_type
        self.n_features_out = n_features_out

    @staticmethod
    def relevance_(X: pd.DataFrame, y: pd.Series, kind: str, target_type: str) -> Any:
        """
        Calculate relevance based on the kind and target_type.
        """
        if kind == "mi":
            if target_type == "discrete":
                rel = mutual_info_classif(X, y)
            elif target_type == "continuous":
                rel = mutual_info_regression(X, y)
            else:
                raise ValueError(f"Unknown target type: {target_type}")
        elif kind == "fs":
            if target_type == "discrete":
                rel, _ = f_classif(X, y)
            elif target_type == "continuous":
                rel, _ = f_regression(X, y)
            else:
                raise ValueError(f"Unknown target type: {target_type}")
        else:
            raise ValueError(f"Unknown kind: {kind}")
        return rel

    @staticmethod
    def redundancy_(X: pd.DataFrame, y: pd.Series, kind: str, target_type: str) -> Any:
        """
        Calculate redundancy based on the kind and target_type.
        """
        if kind == "mi":
            if target_type == "discrete":
                red = mutual_info_classif(X, y)
            elif target_type == "continuous":
                red = mutual_info_regression(X, y)
            else:
                raise ValueError(f"Unknown target type: {target_type}")
        elif kind == "pc":
            red = r_regression(X, y)
        else:
            raise ValueError(f"Unknown kind: {kind}")
        return red

    def mrmr(self, X: pd.DataFrame, y: pd.Series) -> Any:
        """
        Calculate mRMR (minimum Redundancy Maximum Relevance)
        based on the scheme.
        """
        rel = self.relevance_(X, y, kind=self.relevance, target_type=self.target_type)
        mrmr = []
        for i, column in enumerate(X.columns):
            red_sum = self.redundancy_(
                X, y=X[column], kind=self.redundancy, target_type="continuous"
            ).sum()
            if self.scheme == "difference":
                mrmr.append(rel[i] - red_sum / len(X.columns))
            elif self.scheme == "quotient":
                mrmr.append(rel[i] / (red_sum / len(X.columns)))
            else:
                raise ValueError(f"Unknown scheme: {self.scheme}")
        return np.array(mrmr)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> MrmrDF:
        """
        Fit the feature selector.
        """
        self.mrmrs = self.mrmr(X, y)
        self.ranking = np.argsort(self.mrmrs)
        return self

    def transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """
        Transform the input features using the feature selector.
        """
        return X.iloc[:, self.ranking[-self.n_features_out :]]


#
# validate __all__
#
__tracker.validate()
