class Constant:
    BALL = "football"
    QB = "QB"


class Column:
    OBJECT_ID = "id"

    GAME_ID = "game_id"
    FRAME_ID = "frame_id"
    PLAY_ID = "play_id"

    X = "x"
    Y = "y"

    SPEED = "v"

    ACCELERATION = "a"

    TEAM_ID = "team_id"
    POSITION_NAME = "position_name"

    BALL_OWNING_TEAM_ID = "ball_owning_team_id"

    ORIENTATION = "o"
    DIRECTION = "dir"
    HEIGHT_CM = "height_cm"
    WEIGHT_KG = "weight_kg"


class Group:
    BY_FRAME = [Column.GAME_ID, Column.PLAY_ID, Column.FRAME_ID]
    BY_PLAY_BALL_OWNING = [Column.GAME_ID, Column.PLAY_ID, Column.BALL_OWNING_TEAM_ID]
