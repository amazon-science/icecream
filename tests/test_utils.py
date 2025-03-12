# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import pandas as pd
import numpy as np
import pytest

from explain import BernoulliModel, BinaryFunctionModel
from explain import CausalNetwork
from explain import (
    dataframe_to_array,
    power_set,
    list_coalitions,
    noisify,
    denoisify,
)


def test_given_dataframe_then_dataframe_to_array_returns_expected_result():
    a = np.random.random((4, 4))
    df = pd.DataFrame(a, columns=["W", "X", "Y", "Z"])

    assert np.array_equal(dataframe_to_array(df, ["X"]), a[:, 1:2])
    assert np.array_equal(dataframe_to_array(df, ["W", "Y"]), a[:, [0, 2]])

    with pytest.raises(ValueError):
        dataframe_to_array(df, [])


def test_given_set_then_power_set_returns_expected_result():
    s = {0, 1, 2}
    assert power_set(s) == [set(), {0}, {1}, {2}, {0, 1}, {0, 2}, {1, 2}, {0, 1, 2}]


def test_given_causal_network_then_list_coalitions_returns_expected_result():
    X = BernoulliModel()
    Y = BernoulliModel()
    Z = BinaryFunctionModel(
        parent_signature=[BernoulliModel, BernoulliModel],
        f=lambda x, y: np.logical_and(x, y),
    )

    causal_network = CausalNetwork({"X": X, "Y": Y, "Z": (Z, ["X", "Y"])})

    assert list_coalitions(causal_network) == [set(), {"X"}, {"Y"}, {"X", "Y"}]


def test_given_nodes_then_noisify_returns_expected_result():
    nodes = ["X", "Y", "Z"]

    assert noisify(nodes) == ["_X", "_Y", "_Z"]


def test_given_noise_nodes_then_denoisify_returns_expected_result():
    nodes = ["_X", "_Y", "_Z"]

    assert denoisify(nodes) == ["X", "Y", "Z"]
