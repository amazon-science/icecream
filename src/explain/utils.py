# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


# Annotations and typing
from __future__ import annotations
from typing import Collection, Any, Set, List

# Standard modules
from itertools import combinations

# External modules
import numpy as np
import pandas as pd
from scipy.io import arff

# Internal modules
from explain.causal_model import (
    DeterministicCausalModel,
)


# Property keys for networkx graphs
CAUSAL_MODEL = "causal_mechanism"
PARENTS = "parents"
PARENTS_DURING_FIT = "parents_during_fit"


def dataframe_to_array(dataframe: pd.DataFrame, features: List[Any]) -> np.ndarray:
    """Extracts one or more columns from a dataframe and converts the
    result into a numpy array.

    Args:
        dataframe (pd.DataFrame): Dataframe from which the features are extracted
        features (List[Any]): Features to extract

    Returns:
        np.ndarray: Array containing the requested data in the correct order
    """
    if isinstance(features, List):
        if len(features) == 0:
            raise ValueError

        feature_samples = dataframe[features].to_numpy()
        return np.hstack(
            [
                np.vstack(column)
                if isinstance(column[0], np.ndarray)
                else column.reshape((-1, 1))
                for column in feature_samples.T
            ]
        )
    else:
        feature_samples = dataframe[features].to_numpy()
        return (
            np.vstack(feature_samples)
            if isinstance(feature_samples[0], np.ndarray)
            else feature_samples
        )


def series_to_array(series, features):
    return series[list(features)].to_numpy().reshape((1, -1))


def power_set(base_set: Collection[Any]) -> List[Set[Any]]:
    """Lists all subsets of a given set, including the empty set.

    :param base_set:
    :return:
    """
    result = []
    for r in range(len(base_set) + 1):
        size_r_subsets = map(set, combinations(base_set, r))
        result.extend(size_r_subsets)

    return result


def list_coalitions(
    causal_network,
    *,
    exclude_deterministic_nodes: bool = True,
    nodes_to_exclude: Set[Any] = None,
) -> List[Set[Any]]:
    """Generates a list of all coalitions for a given causal network.

    Args:
        causal_network (_type_): Causal network whose nodes are used for the coalitions.
        exclude_deterministic_nodes (bool, optional): If true, only return coalitions
        of nodes which have noise. Defaults to True.
        nodes_to_exclude (Set[Any], optional): Set of nodes which should not appear in
        the coalitions. Defaults to None.

    Returns:
        List[Set[Any]]: List of all coalitions for a given causal network.
    """
    if nodes_to_exclude is None:
        nodes_to_exclude = set()

    coalition_nodes = sorted(
        node
        for node in causal_network.nodes
        if not (
            exclude_deterministic_nodes
            and isinstance(causal_network[node], DeterministicCausalModel)
        )
        and node not in nodes_to_exclude
    )
    return power_set(coalition_nodes)


def noisify(nodes: Collection[Any]) -> List[Any]:
    """Turns a list of node names into a list of corresponding noise names by
    prefixing '_'.

    :param nodes: List of node names.
    :return: List of noise names.
    """
    return [f"_{node}" for node in nodes]


def denoisify(nodes: Collection[Any]) -> List[Any]:
    """Turns a list of noise names into a list of corresponding node names by removing
    the prefix '_'.

    :param nodes: List of noise names.
    :return: List of node names.
    """
    return [node[1:] for node in nodes]


def load_arff(filename: str) -> pd.DataFrame:
    """Loads the content of an .arff file into a dataframe.

    :param filename:
    :return:
    """
    data = arff.loadarff(filename)
    samples = pd.DataFrame(data[0])

    for col in samples:
        samples[col] = samples[col].apply(lambda x: int(x.decode()))

    return samples
