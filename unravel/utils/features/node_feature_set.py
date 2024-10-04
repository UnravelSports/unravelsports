import numpy as np

from .utils import (
    normalize_coords,
    unit_vector,
    normalize_speed,
    normalize_angles,
    normalize_distance,
)


class NodeFeatureSet:
    """
    To manage and store node feature functions configured by user
    """

    def __init__(self):
        self.node_feature_functions = []

    def add_x(self, normed: bool = True):
        """
        Adds a function to calculate the x coordinate to the node feature set.
        If 'normed=True', the function will normalize the x coordinate
        """
        if normed:
            self.node_feature_functions.append(
                ("normalize_x", normalize_coords, ["x", "max_x"])
            )
        else:
            self.node_feature_functions.append(("coord_x", lambda x: x, ["x"]))

        return self

    def add_y(self, normed: bool = True):
        """
        Adds a function to calculate the x coordinate to the node feature set.
        If 'normed=True', the function will normalize the x coordinate
        """
        if normed:
            self.node_feature_functions.append(
                ("normalize_y", normalize_coords, ["y", "max_y"])
            )
        else:
            self.node_feature_functions.append(("coord_y", lambda y: y, ["y"]))

        return self

    def add_velocity(
        self, x: bool = True, y: bool = True, angle: bool = True, normed: bool = True
    ):
        """
        Adds a function to return the x and y unit vectors of the velocity as well as the angle.
        The angle can be normalized and is only calculated if both x and y components of velocity is present.
        """
        if not (x or y):
            print(
                "Warning: No velocity component added. Please add either x or y components"
            )
            return
        if x:
            self.node_feature_functions.append(
                (
                    "unit_velocity_x",
                    lambda velocity: unit_vector(velocity)[0],
                    ["velocity"],
                )
            )
        if y:
            self.node_feature_functions.append(
                (
                    "unit_velocity_y",
                    lambda velocity: unit_vector(velocity)[1],
                    ["velocity"],
                )
            )

        if angle:
            if not (x and y):
                print(
                    "Warning: Angle cannot be calculated because either x or y component was not computed"
                )
            else:
                if normed:
                    self.node_feature_functions.append(
                        (
                            "normalized_velocity_angle",
                            lambda velocity: normalize_angles(
                                np.arctan2(velocity[1], velocity[0])
                            ),
                            ["velocity"],
                        )
                    )
                else:
                    self.node_feature_functions.append(
                        (
                            "velocity_angle",
                            lambda velocity: np.arctan2(velocity[1], velocity[0]),
                            ["velocity"],
                        )
                    )
        return self

    def add_speed(self, normed: bool = True):
        """
        Adds a function that calculates the speed. Can be normalized
        """
        if normed:
            self.node_feature_functions.append(
                ("normalized_speed", normalize_speed, ["speed", "max_speed"])
            )  # Have to round this
        else:
            self.node_feature_functions.append(("speed", lambda s: s, ["speed"]))
        return self

    def add_goal_distance(self, normed: bool = True):
        """
        Adds a function that calculates the distance of the ball/player to the goal. Can be normalized
        """
        if normed:
            self.node_feature_functions.append(
                (
                    "normalized_goal_distance",
                    lambda position, goal_mouth_position, max_dist_to_goal: normalize_distance(
                        np.linalg.norm(position - goal_mouth_position), max_dist_to_goal
                    ),
                    ["position", "goal_mouth_position", "max_dist_to_goal"],
                )
            )
        else:
            self.node_feature_functions.append(
                (
                    "goal_distance",
                    lambda position, goal_mouth_position: np.linalg.norm(
                        position - goal_mouth_position
                    ),
                    ["position", "goal_mouth_position"],
                )
            )
        return self

    def add_goal_angle(self, normed: bool = True):
        """
        Adds a function that calculates the angle of the player to the goal. Can be normalized
        """
        if normed:
            self.node_feature_functions.append(
                ("normed_goal_angle", normalize_angles, ["goal_angle"])
            )
        else:
            self.node_feature_functions.append(
                ("goal_angle", lambda a: a, ["goal_angle"])
            )

        return self

    def add_ball_distance(self, normed: bool = True):
        """
        Adds a function to calculate the distance of the player from the ball. Can be normalized
        """
        if normed:
            self.node_feature_functions.append(
                (
                    "normalized_ball_distance",
                    lambda position, ball_position, max_dist_to_player: normalize_distance(
                        np.linalg.norm(position - ball_position), max_dist_to_player
                    ),
                    ["position", "ball_position", "max_dist_to_player"],
                )
            )
        else:
            self.node_feature_functions.append(
                (
                    "ball_distance",
                    lambda position, ball_position: np.linalg.norm(
                        position - ball_position
                    ),
                    ["position", "ball_position"],
                )
            )
        return self

    def add_ball_angle(self, normed: bool = True):
        """
        Adds a function to calculate the angle of player to the ball. Can be normalized
        """
        if normed:
            self.node_feature_functions.append(
                ("normed_ball_angle", normalize_angles, ["ball_angle"])
            )
        else:
            self.node_feature_functions.append(
                ("ball_angle", lambda a: a, ["ball_angle"])
            )

        return self

    def add_team(self):
        """
        Adds a function that returns 1 if player is on same team but not in possession, 0.1 for all other players, 0.1 if the player is 'missing'
        """
        self.node_feature_functions.append(("team", lambda t: t, ["team"]))
        return self

    def add_potential_reciever(self):
        """
        Adds a function that returns 1 if player is a potential reciever
        """
        self.node_feature_functions.append(
            (
                "potential_reciever",
                lambda potential_receiver, non_potential_receiver_node_value: (
                    1.0 if potential_receiver else non_potential_receiver_node_value
                ),
                ["potential_receiver", "non_potential_receiver_node_value"],
            )
        )
        return self

    def get_features(self):
        return self.node_feature_functions
