from .utils import normalize_coords, coord


class NodeFeatureSet:
    """
    To manage and store node feature functions configured by user
    """

    def __init__(self):
        self.node_feature_functions = []

    def add_x(self, normed: bool = False):
        """
        Adds a function to calculate the x coordinate to the node feature set.
        If 'normed=True', the function will normalize the x coordinate
        """
        if normed:
            print("Added normalize_coords to node_feature_function")
            self.node_feature_functions.append(normalize_coords)
        else:
            print("Added coordinate to node feature function")
            self.node_feature_functions.append(coord)

        return self

    def get_features(self):
        return self.node_feature_functions

