# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import pytest
import pandas as pd
import numpy as np

from explain import BernoulliModel, BinaryFunctionModel, CausalNetwork
from explain.explanations.causal_explanation import (
    exact_explanation_score,
    expected_exact_explanation_scores,
    find_minimum_size_coalitions,
)


def test_given_causal_network_then_explanation_score_returns_expected_result():
    X = BernoulliModel(p=0.5)
    Y = BernoulliModel(p=0.3)
    Z = BinaryFunctionModel(
        parent_signature=[BernoulliModel, BernoulliModel],
        f=lambda parent_values: np.prod(parent_values.astype(int), axis=1).astype(str),
    )

    causal_network = CausalNetwork({"X": X, "Y": Y, "Z": (Z, ["X", "Y"])})
    observation = causal_network.draw_samples(1).iloc[0]

    exact_explanation_score(causal_network, observation, "Z", {"X", "Y"})

    with pytest.raises(ValueError):
        exact_explanation_score(
            causal_network, pd.Series({"X": 0, "Y": 1, "Z": 2}), "Z", {"X", "Y"}
        )

    assert expected_exact_explanation_scores(
        causal_network, observation, "Z", [{"X", "Y", "Z"}]
    )["__s"].values == np.array([1])
    assert expected_exact_explanation_scores(causal_network, observation, "Z", [{}])[
        "__s"
    ].values == np.array([0])


def test_given_causal_network_then_find_minimum_size_coalitions_returns_expected_result():
    X = BernoulliModel(p=0.5)
    Y = BernoulliModel(p=0.3)
    Z = BinaryFunctionModel(
        parent_signature=[BernoulliModel, BernoulliModel],
        f=lambda parent_values: np.prod(parent_values.astype(int), axis=1).astype(str),
    )

    causal_network = CausalNetwork({"X": X, "Y": Y, "Z": (Z, ["X", "Y"])})

    assert set(map(frozenset, find_minimum_size_coalitions(
        causal_network, pd.Series({"X": "0", "Y": "0", "Z": "0"}), "Z"
    ))) == {frozenset({"X"}), frozenset({"Y"})}

    assert find_minimum_size_coalitions(
        causal_network, pd.Series({"X": "1", "Y": "1", "Z": "1"}), "Z"
    ) == [{"X", "Y"}]
