from typing import Dict, List, Literal

import polars as pl

import random

from kloppy.domain import TrackingDataset


def dummy_labels(dataset: TrackingDataset) -> Dict:
    """
    Create dummy labels to feed into GraphNeuralNetworkConverter
    """
    if not isinstance(dataset, TrackingDataset):
        raise TypeError("dataset should be of type TrackingDataset (from kloppy)")

    labels = dict()
    for frame in dataset:
        labels[frame.frame_id] = random.choice([True, False])
    return labels


def dummy_graph_ids(dataset: TrackingDataset) -> Dict:
    """
    Create dummy graph_ids to feed into GraphNeuralNetworkConverter
    """
    if not isinstance(dataset, TrackingDataset):
        raise TypeError("dataset should be of type TrackingDataset (from kloppy)")

    from uuid import uuid4

    graph_ids = dict()
    fake_match_id = str(uuid4())

    for i, frame in enumerate(dataset):
        fake_possession_id = i % 10
        graph_ids[frame.frame_id] = f"{fake_match_id}-{fake_possession_id}"
    return graph_ids


def add_dummy_label_column(
    dataset: pl.DataFrame,
    by: List[str] = ["gameId", "playId", "frameId"],
    column_name: str = "label",
):

    labels = dataset.group_by(by).agg(
        [
            pl.col("a")
            .map_elements(lambda _: random.choice([0, 1]), return_dtype=int)
            .alias(column_name),
        ]
    )
    return dataset.join(labels, on=by, how="left")


def add_graph_id_column(
    dataset: pl.DataFrame,
    by: List[str] = ["gameId", "playId"],
    column_name: str = "graph_id",
):
    return dataset.with_columns([pl.concat_str(by, separator="-").alias(column_name)])
