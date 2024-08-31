from .utils import (
    normalize_x,
    coord_x,
    normalize_y,
    coord_y,
)


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
            self.node_feature_functions.append(normalize_x)
        else:
            self.node_feature_functions.append(coord_x)

        return self
    
    def add_y(self, normed: bool = False):
        """
        Adds a function to calculate the x coordinate to the node feature set.
        If 'normed=True', the function will normalize the x coordinate
        """
        if normed:
            self.node_feature_functions.append(normalize_y)
        else:
            self.node_feature_functions.append(coord_y)

        return self

    def get_features(self):
        return self.node_feature_functions

