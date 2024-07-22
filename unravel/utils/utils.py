from typing import Dict

from kloppy.domain import TrackingDataset


def dummy_labels(dataset: TrackingDataset) -> Dict:
    """
    Create dummy labels to feed into GraphNeuralNetworkConverter
    """
    if not isinstance(dataset, TrackingDataset):
        raise TypeError("dataset should be of type TrackingDataset (from kloppy)")
    import random

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
