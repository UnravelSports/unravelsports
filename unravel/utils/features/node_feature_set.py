import numpy as np

from .utils import (
    normalize_coords,
    coord,
    unit_vector,
    normalize_speed,
    normalize_angles,
    normalize_distance
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
    
    def add_velocity(self, x: bool = True, y: bool = True, angle: bool = True, normed: bool = False):
        if not (x or y):
            print("Warning: No velocity component added. Please add either x or y components")
            return
        if x:
            self.node_feature_functions.append(('unit_velocity_x', lambda velocity: unit_vector(velocity)[0], ['velocity']))
        if y:
            self.node_feature_functions.append(('unit_velocity_y', lambda velocity: unit_vector(velocity)[1], ['velocity']))
        
        if angle:
            if not (x and y):
                print("Warning: Angle cannot be calculated because either x or y component was not computed")
            else:
                if normed:
                    self.node_feature_functions.append(('normalized_velocity_angle', lambda velocity: normalize_angles(np.arctan2(velocity[1], velocity[0])), ['velocity']))
                else:
                    self.node_feature_functions.append(('velocity_angle', lambda velocity: np.arctan2(velocity[1], velocity[0]), ['velocity']))
        return self
    
    def add_speed(self, normed: bool = True):
        if normed:
            self.node_feature_functions.append(('normalized_speed', normalize_speed, ['speed', 'max_speed'])) #Have to round this
        else:
            self.node_feature_functions.append(('speed', coord, ['speed']))
        return self
    
    def add_goal_metric(self, normed: bool = True):
        if normed:
            self.node_feature_functions.append(('normalized_goal_distance', 
                                                lambda position, goal_mouth_position, max_dist_to_goal: normalize_distance(
                                                np.linalg.norm(position - goal_mouth_position),
                                                max_dist_to_goal)
                                                , ['position', 'goal_mouth_position', 'max_dist_to_goal']))
        else:
            self.node_feature_functions.append(('goal_distance', lambda position, goal_mouth_position: np.linalg.norm(position - goal_mouth_position), ['position', 'goal_mouth_position']))
        
            

    def get_features(self):
        return self.node_feature_functions

