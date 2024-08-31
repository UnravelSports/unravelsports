from .utils import (
    normalize_coords,
    coord,
    unit_vector,
    normalize_speed,
    normalize_angles
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
            self.node_feature_functions.append(('normalize_x', normalize_coords, ['x', 'max_x']))
        else:
            self.node_feature_functions.append(('coord_x', coord, ['x']))

        return self
    
    def add_y(self, normed: bool = False):
        """
        Adds a function to calculate the x coordinate to the node feature set.
        If 'normed=True', the function will normalize the x coordinate
        """
        if normed:
            self.node_feature_functions.append(('normalize_y', normalize_coords, ['y', 'max_y']))
        else:
            self.node_feature_functions.append(('coord_y', coord, ['y']))

        return self
    
    def add_velocity(self, x: bool = True, y: bool = True):
        if not (x or y):
            print("Warning: No velocity component added. Please add either x or y components")
            return
        if x:
            self.node_feature_functions.append(('unit_velocity_x', lambda velocity: unit_vector(velocity)[0], ['velocity']))
        if y:
            self.node_feature_functions.append(('unit_velocity_y', lambda velocity: unit_vector(velocity)[1], ['velocity']))
        return self
    
    def add_speed(self, normed: bool = True):
        if normed:
            self.node_feature_functions.append(('normalized_speed', normalize_speed, ['speed', 'max_speed']))
        else:
            self.node_feature_functions.append(('speed', coord, ['speed']))
        return self

    def get_features(self):
        return self.node_feature_functions

