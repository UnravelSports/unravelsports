class Constant:
    BALL = "football"
    QB = "QB"


class Column:
    OBJECT_ID = "nflId"

    GAME_ID = "gameId"
    FRAME_ID = "frameId"
    PLAY_ID = "playId"

    X = "x"
    Y = "y"

    ACCELERATION = "a"
    SPEED = "s"
    ORIENTATION = "o"
    DIRECTION = "dir"
    TEAM = "team"
    CLUB = "club"
    OFFICIAL_POSITION = "officialPosition"
    POSSESSION_TEAM = "possessionTeam"
    HEIGHT_CM = "height_cm"
    WEIGHT_KG = "weight_kg"


class Group:
    BY_FRAME = [Column.GAME_ID, Column.PLAY_ID, Column.FRAME_ID]
    BY_PLAY_POSSESSION_TEAM = [Column.GAME_ID, Column.PLAY_ID, Column.POSSESSION_TEAM]
