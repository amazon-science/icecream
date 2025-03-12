# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


# Annotations and typing
from __future__ import annotations
from typing import TypeVar, Generic, Optional, Dict, Any, Union, Tuple, List, Iterator

# External modules
import networkx as nx
import pandas as pd
from tqdm import tqdm

# Internal modules
from explain.causal_model import (
    CausalModel,
    UnconditionalCausalModel,
    ConditionalCausalModel,
    FiniteNoiseCausalModel,
    DeterministicCausalModel,
    NoiseCausalModel,
    FunctionCausalModel,
    NoiseFunctionCausalModel,
)
from explain.utils import (
    CAUSAL_MODEL,
    PARENTS,
    PARENTS_DURING_FIT,
    dataframe_to_array,
    noisify,
    series_to_array,
)


T = TypeVar("T", bound=CausalModel)


class CausalNetwork(Generic[T]):
    """Represents a causal network, i.e., a causal graph together with causal models
    for each node. The causal models must be of type T or a subclass of T.
    """

    def __init__(
        self, causal_models: Optional[Dict[Any, Union[T, Tuple[T, List[Any]]]]] = None
    ):
        """
        :param causal_models: A dictionary whose keys are the names of the causal
        models, and whose values are either
        - the causal model itself, if it has no parents, or
        - a tuple of (causal_model, list_of_parent_names) if it has parents.
        From this information, the causal graph is built.
        """
        self._graph = nx.DiGraph()

        nx.set_node_attributes(self.graph, None, CAUSAL_MODEL)
        nx.set_node_attributes(self.graph, None, PARENTS)

        if causal_models:
            for node, data in causal_models.items():
                if isinstance(data, CausalModel):
                    self.set_causal_model(node, data, validate=False)
                else:
                    self.set_causal_model(node, *data, validate=False)

            self.validate()

        # Dummy fit for compatibility with gcm.ProbabilisticCausalModel
        for node, _ in self:
            self._graph.nodes[node][PARENTS_DURING_FIT] = (
                sorted(self.parents(node)) if self.parents(node) else []
            )

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} nodes={self.nodes}"

    def __getitem__(self, item) -> T:
        if item not in self._graph.nodes:
            raise ValueError(f"Node {item} can not be found in the given graph!")

        return self._graph.nodes[item][CAUSAL_MODEL]

    def __iter__(self) -> Iterator:
        """Returns an iterator for the nodes of the graph in topological order."""
        return (
            {node: self[node] for node in nx.topological_sort(self._graph)}
            .items()
            .__iter__()
        )

    def set_causal_model(
        self,
        node: Any,
        model: T,
        parents: Optional[List[Any]] = None,
        *,
        validate: bool = True,
    ) -> None:
        if node in self._graph:
            self._graph.remove_edges_from(
                (parent, node) for parent in self._graph.predecessors(node)
            )

        if isinstance(model, ConditionalCausalModel):
            self._graph.add_edges_from((parent, node) for parent in parents or [])

        self._graph.add_node(node, **{CAUSAL_MODEL: model, PARENTS: parents})

        if validate:
            self.validate()

    def fit(self, data: pd.DataFrame) -> None:
        self.validate()

        progress_bar = tqdm(
            self._graph.nodes, desc="Fitting causal models", position=0, leave=True
        )
        for node in progress_bar:
            if node not in data:
                raise RuntimeError(
                    f"Could not find data for node {node} in the given training data!"
                )

            progress_bar.set_description(f"Fitting causal mechanism of node {node}")

            self._fit_causal_model_of_target(node, data)

    def _fit_causal_model_of_target(
        self, target_node: Any, training_data: pd.DataFrame
    ) -> None:
        causal_model = self[target_node]

        if isinstance(causal_model, UnconditionalCausalModel):
            causal_model.fit(X=dataframe_to_array(training_data, target_node))
        else:
            causal_model.fit(
                X=dataframe_to_array(
                    training_data, self._graph.nodes[target_node][PARENTS]
                ),
                Y=dataframe_to_array(training_data, target_node),
            )

    def draw_samples(self, num_samples: int) -> pd.DataFrame:
        self.validate()

        samples = pd.DataFrame(index=range(num_samples), columns=list(self.nodes))

        for node, causal_model in self:
            if isinstance(causal_model, UnconditionalCausalModel):
                samples[node] = list(causal_model.draw_samples(num_samples))
            else:
                samples[node] = list(
                    causal_model.draw_samples(
                        dataframe_to_array(samples, self.parents(node))
                    )
                )

        return samples

    def draw_samples_with_noise(self, num_samples: int) -> pd.DataFrame:
        self.validate()

        samples = pd.DataFrame(
            index=range(num_samples), columns=list(self.nodes) + noisify(self.nodes)
        )

        for node, causal_model in self:
            if isinstance(causal_model, NoiseCausalModel):
                samples[f"_{node}"] = list(causal_model.draw_noise_samples(num_samples))
                samples[node] = list(causal_model.evaluate(samples[f"_{node}"]))
            elif isinstance(causal_model, FunctionCausalModel):
                samples[f"_{node}"] = 0
                samples[node] = list(
                    causal_model.draw_samples(
                        dataframe_to_array(samples, self.parents(node))
                    )
                )
            elif isinstance(causal_model, NoiseFunctionCausalModel):
                samples[f"_{node}"] = list(causal_model.draw_noise_samples(num_samples))
                samples[node] = list(
                    causal_model.evaluate(
                        dataframe_to_array(samples, self.parents(node)),
                        samples[f"_{node}"],
                    )
                )
            else:
                raise ValueError

        return samples

    def draw_noise_samples(self, num_samples: int) -> pd.DataFrame:
        self.validate()

        samples = pd.DataFrame(index=range(num_samples), columns=noisify(self.nodes))

        for node, causal_model in self:
            if isinstance(causal_model, NoiseCausalModel) or isinstance(
                causal_model, NoiseFunctionCausalModel
            ):
                samples[f"_{node}"] = list(causal_model.draw_noise_samples(num_samples))
            elif isinstance(causal_model, FunctionCausalModel):
                samples[f"_{node}"] = 0
            else:
                raise ValueError

        return samples

    def from_noise(self, noise_samples: pd.DataFrame) -> pd.DataFrame:
        self.validate()

        samples = pd.DataFrame(
            noise_samples, columns=list(self.nodes) + noisify(self.nodes)
        )

        for node, causal_model in self:
            if isinstance(causal_model, NoiseCausalModel):
                samples[node] = list(causal_model.evaluate(samples[f"_{node}"]))
            elif isinstance(causal_model, FunctionCausalModel):
                samples[node] = list(
                    causal_model.draw_samples(
                        dataframe_to_array(samples, self.parents(node))
                    )
                )
            elif isinstance(causal_model, NoiseFunctionCausalModel):
                samples[node] = list(
                    causal_model.evaluate(
                        dataframe_to_array(samples, self.parents(node)),
                        samples[f"_{node}"],
                    )
                )
            else:
                raise ValueError

        return samples

    def draw_interventional_samples(self, interventions: pd.DataFrame) -> pd.DataFrame:
        self.validate()

        samples = pd.DataFrame(
            interventions, columns=list(self.nodes) + noisify(self.nodes)
        )

        for node, causal_model in self:
            if isinstance(causal_model, NoiseCausalModel):
                samples[f"_{node}"].fillna(
                    pd.Series(
                        causal_model.draw_noise_samples(len(samples)),
                        index=samples.index,
                    ),
                    inplace=True,
                )
                samples[node].fillna(
                    pd.Series(
                        causal_model.evaluate(samples[f"_{node}"]), index=samples.index
                    ),
                    inplace=True,
                )
            elif isinstance(causal_model, FunctionCausalModel):
                samples[f"_{node}"] = 0
                samples[node].fillna(
                    pd.Series(
                        causal_model.draw_samples(
                            dataframe_to_array(samples, self.parents(node))
                        ),
                        index=samples.index,
                    ),
                    inplace=True,
                )
            elif isinstance(causal_model, NoiseFunctionCausalModel):
                samples[f"_{node}"].fillna(
                    pd.Series(
                        causal_model.draw_noise_samples(len(samples)),
                        index=samples.index,
                    ),
                    inplace=True,
                )
                samples[node].fillna(
                    pd.Series(
                        causal_model.evaluate(
                            dataframe_to_array(samples, self.parents(node)),
                            samples[f"_{node}"],
                        ),
                        index=samples.index,
                    ),
                    inplace=True,
                )
            else:
                raise ValueError

        return samples

    def get_node_value(self, target_node: Any, noise_values: Dict[Any, Any]) -> Any:
        """Returns the value of a given node, if all required noise values are provided.
        :param target_node:
        :param noise_values:
        :return:
        """
        values = pd.Series(noise_values, index=list(self.nodes) + noisify(self.nodes))

        for node, causal_model in self:
            if isinstance(causal_model, NoiseCausalModel):
                values[node] = values[f"_{node}"]
            elif isinstance(causal_model, FunctionCausalModel):
                values[node] = causal_model.evaluate(
                    parent_samples=series_to_array(values, self.parents(node))
                )
            elif isinstance(causal_model, NoiseFunctionCausalModel):
                values[node] = causal_model.evaluate(
                    parent_samples=series_to_array(values, self.parents(node)),
                    noise_samples=series_to_array(values, [f"_{node}"]),
                )
            else:
                raise ValueError

            if node == target_node:
                return values[target_node]

    @property
    def nodes(self):
        return nx.topological_sort(self._graph)

    def parents(self, node) -> List[Any]:
        return self._graph.nodes[node][PARENTS]

    def has_finite_noise(self) -> bool:
        """Determines if all noise domains (and therefore their product) are finite.

        Returns:
            bool: True if all nodes have finite noise domains, otherwise False.
        """
        return all(
            isinstance(self[node], DeterministicCausalModel)
            or isinstance(self[node], FiniteNoiseCausalModel)
            for node in self._graph.nodes
        )

    def validate(self) -> None:
        """Validates the matching between each node's parent list and its predecessors
        in the graph. In particular, two things are checked:
        - If the node is an UnconditionalCausalModel, it cannot have any parents.
        - If the node is a ConditionalCausalModel, its parent signature needs to match
        the types of the predecessor node in the order given by `node.parents`.
        If any of these requirements is not met, throws an AssertionError.
        """
        for node in self._graph.nodes:
            causal_model = self[node]

            if isinstance(causal_model, UnconditionalCausalModel):
                assert self._graph.in_degree(node) == 0
            else:
                assert self._graph.in_degree(node) > 0 and set(
                    self.parents(node)
                ) == set(self._graph.predecessors(node))
                assert all(
                    isinstance(self[parent], parent_type)
                    for parent, parent_type in zip(
                        self.parents(node), causal_model.parent_signature
                    )
                ), (
                    f"Parent signature {causal_model.parent_signature} of node {node} \
                          does not match the parent types in the network "
                    f"({[self[parent] for parent in self.parents(node)]}!"
                )

    def draw(self, **kwargs) -> None:
        node_colors = [
            "#00ff00" if isinstance(self[node], UnconditionalCausalModel) else "#0000ff"
            for node in self.nodes
        ]
        nx.draw(
            self._graph,
            with_labels=True,
            pos=nx.circular_layout(self._graph),
            node_size=2000,
            font_size=8,
            node_color=node_colors,
        )

    # Compatability methods for gcm.ProbabilisticCausalModel
    def set_causal_mechanism(self, node: Any, mechanism) -> None:
        self.set_causal_model(node, mechanism, self._graph.predecessors(node))

    def causal_mechanism(self, node: Any) -> CausalModel:
        return self[node]

    def clone(self) -> CausalNetwork:
        raise NotImplementedError

    @property
    def graph(self) -> nx.DiGraph:
        return self._graph
