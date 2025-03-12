# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


# Annotations and typing
from __future__ import annotations
from typing import Any, Set, Optional, Callable

# Standard modules
import math
import itertools

# External modules
import pandas as pd
import numpy as np
from joblib import Parallel, delayed


def estimate_explanation_score(
    ground_distribution: pd.DataFrame,
    observation: pd.Series,
    target: Any,
    classifier: Callable,
    coalition: Set[Any],
    *,
    number_of_samples: int = 1_000,
    verbose: bool = False,
) -> float:
    """Calculates the explanation score of a specific coalition."""
    base_samples = ground_distribution.sample(n=number_of_samples, replace=True)
    base_samples[target] = classifier(base_samples.drop(target, axis=1))

    interventional_samples = ground_distribution.sample(
        n=number_of_samples, replace=True
    )
    interventional_samples[list(coalition)] = observation[list(coalition)]
    interventional_samples[target] = classifier(
        interventional_samples.drop(target, axis=1)
    )

    base_probability = (base_samples[target] == observation[target]).mean()
    interventional_probability = (
        interventional_samples[target] == observation[target]
    ).mean()

    if base_probability == 0.0:
        raise ValueError(
            "The explanation score is not defined if the base probability is zero!"
        )

    if interventional_probability == 0.0:
        raise ValueError(
            "The explanation score is not defined if the interventional probability"
            " is zero!"
        )

    score = 1.0 - math.log(interventional_probability) / math.log(base_probability)

    if verbose:
        print(f"{base_probability=}, {interventional_probability=} => {score=}")

    return score


def _estimate_interventional_probability(
    ground_distribution: pd.DataFrame,
    observation: pd.Series,
    target: Any,
    classifier: Callable,
    coalition: Set[Any],
) -> float:
    """Estimates the interventional probability of a specific coalition."""
    interventional_features = list(coalition)

    interventional_samples = ground_distribution.copy()
    interventional_samples[interventional_features] = observation[
        interventional_features
    ]
    interventional_samples[target] = classifier(interventional_samples)

    interventional_probability = (
        interventional_samples[target] == observation[target]
    ).mean()

    return interventional_probability


def estimate_explanation_scores(
    ground_distribution,
    observation,
    target,
    classifier,
    coalitions,
    *,
    number_of_samples=1000,
    number_of_parallel_jobs=-1,
    verbose=False,
):
    # The base probability is the same for all coalitions and is therefore only
    # calculated once
    base_samples = ground_distribution.sample(n=number_of_samples, replace=True)
    base_samples[target] = classifier(base_samples.drop(target, axis=1))

    base_probability = (base_samples[target] == observation[target]).mean()
    if base_probability == 0.0:
        print("Base probability is zero!")
        return

    # Define parallelizable function for joblib
    def estimate_interventional_probability(coalition):
        return _estimate_interventional_probability(
            base_samples, observation, target, classifier, coalition
        )

    estimated_interventional_probabilities = pd.Series(
        Parallel(n_jobs=number_of_parallel_jobs)(
            delayed(estimate_interventional_probability)(coalition)
            for coalition in coalitions
        ),
        index=coalitions,
    )
    estimated_explanation_scores = 1.0 - np.log(
        estimated_interventional_probabilities
    ) / np.log(base_probability)

    return estimated_explanation_scores


def find_minimum_size_coalitions(
    ground_distribution: pd.DataFrame,
    observation: pd.Series,
    target: Any,
    classifier: Callable,
    *,
    number_of_samples: int = 1000,
    minimum_coalition_size: int = 1,
    maximum_coalition_size: Optional[int] = None,
    number_of_parallel_jobs: int = 10,
    threshold: float = 0.998,
    verbose: bool = False,
) -> Set[frozenset]:
    """ """
    coalition_features = set(ground_distribution.columns) - {target}

    if maximum_coalition_size is None:
        maximum_coalition_size = len(coalition_features)

    # The base probability is the same for all coalitions and is therefore only
    # calculated once
    base_samples = ground_distribution.sample(n=number_of_samples, replace=True)
    base_samples[target] = classifier(base_samples.drop(target, axis=1))

    base_probability = (base_samples[target] == observation[target]).mean()
    if base_probability == 0.0:
        print("Base probability is zero!")
        return

    # Define parallelizable function for joblib
    def estimate_interventional_probability(coalition):
        return _estimate_interventional_probability(
            base_samples, observation, target, classifier, coalition
        )

    # Look at coalitions by increasing number of features; if at least one coalition is
    # found, return
    for coalition_size in range(
        minimum_coalition_size, min(len(coalition_features), maximum_coalition_size + 1)
    ):
        coalitions = list(
            map(frozenset, itertools.combinations(coalition_features, coalition_size))
        )

        if verbose:
            print(
                f"Searching coalitions with {coalition_size} features "
                f"({len(coalitions)} coalitions)..."
            )

        # Compute all interventional probabilities in parallel
        estimated_interventional_probabilities = pd.Series(
            Parallel(n_jobs=number_of_parallel_jobs)(
                delayed(estimate_interventional_probability)(coalition)
                for coalition in coalitions
            ),
            index=coalitions,
        )
        estimated_explanation_scores = 1.0 - np.log(
            estimated_interventional_probabilities
        ) / np.log(base_probability)

        # if verbose:
        #   df = pd.DataFrame(estimated_explanation_scores)
        #   df['base'] = base_probability
        #   df['interventional'] = estimated_interventional_probabilities

        #   print(df)

        # Filter for coalitions with explanation score >= threshold
        full_explanation_coalitions = estimated_explanation_scores.index[
            estimated_explanation_scores >= threshold
        ]

        # Return if there is at least one such coalition
        if len(full_explanation_coalitions) > 0:
            return set(full_explanation_coalitions)

    return set()
