class Constant:
    BALL = "ball"


class Column:
    BALL_OWNING_TEAM_ID = "ball_owning_team_id"
    BALL_OWNING_PLAYER_ID = "ball_owning_player_id"
    IS_BALL_CARRIER = "is_ball_carrier"
    PERIOD_ID = "period_id"
    TIMESTAMP = "timestamp"
    BALL_STATE = "ball_state"
    FRAME_ID = "frame_id"
    GAME_ID = "game_id"
    TEAM_ID = "team_id"
    OBJECT_ID = "id"
    POSITION_NAME = "position_name"

    X = "x"
    Y = "y"
    Z = "z"

    SPEED = "v"
    VX = "vx"
    VY = "vy"
    VZ = "vz"

    ACCELERATION = "a"
    AX = "ax"
    AY = "ay"
    AZ = "az"


class Group:
    BY_FRAME = [Column.GAME_ID, Column.PERIOD_ID, Column.FRAME_ID]
    BY_FRAME_TEAM = [Column.GAME_ID, Column.PERIOD_ID, Column.FRAME_ID, Column.TEAM_ID]
    BY_OBJECT_PERIOD = [Column.OBJECT_ID, Column.PERIOD_ID]
    BY_TIMESTAMP = [Column.GAME_ID, Column.PERIOD_ID, Column.FRAME_ID, Column.TIMESTAMP]
