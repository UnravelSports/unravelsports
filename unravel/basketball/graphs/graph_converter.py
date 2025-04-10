import numpy as np
from scipy.sparse import csr_matrix

class BasketballGraphConverter:
    """
    Converts NBA tracking data into graph representations.
    """
    def __init__(self, dataset, settings, pitch_dimensions):
        # dataset: instance of BasketballDataset; settings: BasketballGraphSettings instance;
        # pitch_dimensions: BasketballPitchDimensions instance.
        self.dataset = dataset
        self.settings = settings
        self.pitch_dimensions = pitch_dimensions
        self.graph_frames = []

    def _normalize_coordinates(self, x, y):
        # Normalize coordinates based on court dimensions if enabled.
        if self.settings.normalize_coordinates:
            norm_x = x / self.pitch_dimensions.court_length
            norm_y = y / self.pitch_dimensions.court_width
            return norm_x, norm_y
        return x, y

    def _compute_node_features(self, frame_records):
        # Compute node features (normalized x, y positions) and extract team info.
        features = [[*self._normalize_coordinates(rec['x'], rec['y'])] for rec in frame_records]
        teams = [rec['team'] for rec in frame_records]
        return np.array(features), teams

    def _compute_adjacency(self, teams):
        # Vectorized computation of the adjacency matrix.
        teams_arr = np.array(teams)
        A = (teams_arr[:, None] == teams_arr[None, :]).astype(np.float64)
        if not self.settings.self_loop_ball:
            np.fill_diagonal(A, 0)
        return csr_matrix(A)

    def _compute_edge_features(self, node_features):
        # Vectorized Euclidean distance calculation between all node pairs.
        diff = node_features[:, np.newaxis, :] - node_features[np.newaxis, :, :]
        dist_matrix = np.linalg.norm(diff, axis=2)
        return dist_matrix.flatten()[:, np.newaxis]

    def convert(self):
        """
        Groups the dataset by frame and converts each frame into a graph.
        Returns a list of dicts, each with keys 'id', 'x' (node features), 
        'a' (adjacency matrix), and 'e' (edge features).
        """
        if self.dataset.data is None:
            raise ValueError("Dataset not loaded; call load() first.")
        df = self.dataset.data.to_pandas()
        grouped = df.groupby("frame_id")
        for frame_id, group in grouped:
            records = group.to_dict(orient="records")
            node_features, teams = self._compute_node_features(records)
            A = self._compute_adjacency(teams)
            E = self._compute_edge_features(node_features)
            graph_frame = {"id": frame_id, "x": node_features, "a": A, "e": E}
            self.graph_frames.append(graph_frame)
        return self.graph_frames
