# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import pytest
import pandas as pd
import numpy as np

from explain.explanations.model_explanation import (
    estimate_explanation_score,
    find_minimum_size_coalitions,
)


def test_given_model_then_explanation_score_returns_expected_result():
    def model(feature_values):
        return np.prod(feature_values.astype(int), axis=1).astype(str)

    samples = pd.DataFrame(
        np.random.choice(["0", "1"], size=(100, 2)), columns=["X", "Y"]
    )

    samples["Z"] = model(samples.to_numpy())

    estimate_explanation_score(samples, samples.iloc[0], "Z", model, {"X", "Y"})

    with pytest.raises(ValueError):
        estimate_explanation_score(
            samples, pd.Series({"X": 0, "Y": 1, "Z": 2}), "Z", model, {"X", "Y"}
        )

    assert (
        estimate_explanation_score(
            samples,
            samples.iloc[0],
            "Z",
            model,
            {"X", "Y", "Z"},
            number_of_samples=10_000,
        )
        == 1
    )
    assert estimate_explanation_score(
        samples, samples.iloc[0], "Z", model, {}, number_of_samples=10_000
    ) == pytest.approx(0, abs=0.1)

    estimate_explanation_score(
        samples.astype(str), samples.iloc[0].astype(str), "Z", model, {"X", "Y"}
    )


def test_given_model_then_find_minimum_size_coalitions_returns_expected_result():
    def model(feature_values):
        return np.prod(feature_values.astype(int), axis=1).astype(str)

    samples = pd.DataFrame(
        np.random.choice(["0", "1"], size=(100, 2)), columns=["X", "Y"]
    )

    samples["Z"] = model(samples.to_numpy())

    find_minimum_size_coalitions(samples, samples.iloc[0], "Z", model)
