class BasketballGraphSettings:
    """
    Configuration settings for converting NBA tracking data into graph representations.
    """
    def __init__(
        self,
        self_loop_ball: bool = True,
        adjacency_matrix_connect_type: str = "ball",
        adjacency_matrix_type: str = "split_by_team",
        label_type: str = "binary",
        max_player_speed: float = 20.0,  # unit: feet per second (adjust as needed)
        max_ball_speed: float = 30.0,    # unit: feet per second (adjust as needed)
        defending_team_node_value: float = 0.0,
        attacking_team_node_value: float = 1.0,
        normalize_coordinates: bool = True,
        verbose: bool = False,
    ):
        self.self_loop_ball = self_loop_ball
        self.adjacency_matrix_connect_type = adjacency_matrix_connect_type
        self.adjacency_matrix_type = adjacency_matrix_type
        self.label_type = label_type
        self.max_player_speed = max_player_speed
        self.max_ball_speed = max_ball_speed
        self.defending_team_node_value = defending_team_node_value
        self.attacking_team_node_value = attacking_team_node_value
        self.normalize_coordinates = normalize_coordinates
        self.verbose = verbose

    def as_dict(self) -> dict:
        """Return all settings as a dictionary."""
        return {
            "self_loop_ball": self.self_loop_ball,
            "adjacency_matrix_connect_type": self.adjacency_matrix_connect_type,
            "adjacency_matrix_type": self.adjacency_matrix_type,
            "label_type": self.label_type,
            "max_player_speed": self.max_player_speed,
            "max_ball_speed": self.max_ball_speed,
            "defending_team_node_value": self.defending_team_node_value,
            "attacking_team_node_value": self.attacking_team_node_value,
            "normalize_coordinates": self.normalize_coordinates,
            "verbose": self.verbose,
        }
