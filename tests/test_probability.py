# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np

from explain import (
    BernoulliModel,
    BinaryFunctionModel,
    CausalNetwork,
    noise_probability_table,
    recover_noise_distribution,
)


def test_given_causal_network_then_noise_probability_table_returns_expected_result():
    X = BernoulliModel(p=0.5)
    Y = BernoulliModel(p=0.3)
    Z = BinaryFunctionModel(
        parent_signature=[BernoulliModel, BernoulliModel],
        f=lambda parent_values: np.prod(parent_values.astype(int), axis=1),
    )

    causal_network = CausalNetwork({"X": X, "Y": Y, "Z": (Z, ["X", "Y"])})

    assert len(noise_probability_table(causal_network)) == 4


def test_given_causal_network_then_recover_noise_distribution_returns_expected_result():
    X = BernoulliModel(p=0.5)
    Y = BernoulliModel(p=0.3)
    Z = BinaryFunctionModel(
        parent_signature=[BernoulliModel, BernoulliModel],
        f=lambda parent_values: np.prod(parent_values.astype(int), axis=1),
    )

    causal_network = CausalNetwork({"X": X, "Y": Y, "Z": (Z, ["X", "Y"])})

    print(recover_noise_distribution(causal_network, {'X': 1}))
