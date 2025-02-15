import polars as pl


def remove_plays(data):
    """
    Because we are trying to predict is a pass is being thrown from the pre-snap paterns
    We remove plays that are pass or run
    """
    remove_plays = (
        data.filter(
            ((pl.col("frameType") == "BEFORE_SNAP") | (pl.col("frameType") == "SNAP"))
            & (
                pl.col("event").is_in(
                    [
                        "field_goal_play",
                        "pass_forward",
                        "timeout_away",
                        "timeout_home",
                        "snap_direct",
                    ]
                )
            )
        )
        .select(["gameId", "playId"])
        .unique()
    )
    data = (
        data.join(remove_plays, on=["gameId", "playId"], how="anti")
        .filter(
            (pl.col("frameType") == "BEFORE_SNAP") | (pl.col("frameType") == "SNAP")
        )
        .sort(by=["gameId", "playId", "frameId", "nflId"])
        .with_columns(
            pl.col("event")
            .fill_null(strategy="forward")
            .over(["gameId", "playId", "nflId"])
            .alias("event")
        )
        .filter(
            ~pl.col("event").is_in(["huddle_break_offense", "huddle_start_offense"])
        )
        .filter(pl.col("frameType") != "SNAP")
    )
    return data


def plays_variables(plays, games):
    data = plays.select(
        [
            "gameId",
            "playId",
            "possessionTeam",
            "quarter",
            "down",
            "yardsToGo",
            "yardlineNumber",
            "gameClock",
            "preSnapHomeScore",
            "preSnapVisitorScore",
            "preSnapHomeTeamWinProbability",
            "passResult",
            "passLength",
            "prePenaltyYardsGained",
            "yardsGained",
        ]
    ).join(games.select(["gameId", "homeTeamAbbr", "visitorTeamAbbr"]), on="gameId")
    print(data.columns)
    data = data.with_columns(
        [
            pl.when(pl.col("possessionTeam") == pl.col("homeTeamAbbr"))
            .then(True)
            .otherwise(False)
            .alias("isHome"),
            pl.when(pl.col("possessionTeam") == pl.col("homeTeamAbbr"))
            .then(pl.col("preSnapHomeScore") - pl.col("preSnapVisitorScore"))
            .otherwise(pl.col("preSnapVisitorScore") - pl.col("preSnapHomeScore"))
            .alias("preSnapScoreDiff"),
            pl.when(pl.col("possessionTeam") == pl.col("homeTeamAbbr"))
            .then(pl.col("preSnapHomeTeamWinProbability"))
            .otherwise(1 - pl.col("preSnapHomeTeamWinProbability"))
            .alias("preSnapTeamWinProbability"),
            pl.col("gameClock")
            .str.split(":")
            .list.get(0)
            .cast(pl.Int32)
            .alias("quarterMinute"),
            pl.col("gameClock")
            .str.split(":")
            .list.get(1)
            .cast(pl.Int32)
            .alias("quarterSecond"),
        ]
    )
    data = data.with_columns(
        [
            pl.when(pl.col("quarter") <= 4)
            .then(
                3600
                - (
                    (pl.col("quarter").clip(lower_bound=1) - 1) * 900
                    + (pl.col("quarterMinute") * 60 + pl.col("quarterSecond"))
                )
            )
            .otherwise(600 - (pl.col("quarterMinute") * 60 + pl.col("quarterSecond")))
            .alias("gameSecondsLeft"),
            pl.when(pl.col("quarter") <= 4)
            .then(900 - (pl.col("quarterMinute") * 60 + pl.col("quarterSecond")))
            .otherwise(600 - (pl.col("quarterMinute") * 60 + pl.col("quarterSecond")))
            .alias("quarterSecondsLeft"),
            pl.col("passResult").is_not_null().alias("isPass"),
            pl.col("yardsToGo").alias("yardsToGoText"),
            (
                pl.col("visitorTeamAbbr").cast(str)
                + " "
                + pl.col("preSnapVisitorScore").cast(str)
                + " - "
                + pl.col("homeTeamAbbr").cast(str)
                + " "
                + pl.col("preSnapHomeScore").cast(str)
            ).alias("scoreBoard"),
        ]
    )
    print(data.columns)
    data = data.with_columns(
        [
            pl.col("yardsToGo") / 50.0,
            pl.col("preSnapScoreDiff") / 100.0,
            pl.when(pl.col("quarter") <= 4)
            .then(pl.col("quarterMinute") / 15.0)
            .otherwise(pl.col("quarterMinute") / 10.0)
            .alias("quarterMinute"),
            pl.col("quarterSecond") / 60.0,
            pl.when(pl.col("quarter") <= 4)
            .then(pl.col("gameSecondsLeft") / 3600)
            .otherwise(pl.col("gameSecondsLeft") / 600)
            .alias("gameSecondsLeft"),
            pl.when(pl.col("quarter") <= 4)
            .then((pl.col("quarterMinute") * 60 + pl.col("quarterSecond")) / 900)
            .otherwise((pl.col("quarterMinute") * 60 + pl.col("quarterSecond")) / 600)
            .alias("quarterSecondsLeft"),
            pl.col("yardlineNumber") / 50,
            (pl.col("prePenaltyYardsGained") / 100.0).alias(
                "prePenaltyYardsGainedNorm"
            ),
        ]
    )

    data = data.drop(
        [
            "gameClock",
            "preSnapHomeScore",
            "preSnapVisitorScore",
            "preSnapHomeTeamWinProbability",
            "passResult",
            "passLength",
            "homeTeamAbbr",
            "possessionTeam",
        ]
    )

    y_columns = [
        "prePenaltyYardsGained",
        "yardsGained",
        "isPass",
        "prePenaltyYardsGainedNorm",
    ]

    x_columns = [
        "quarter",
        "down",
        "yardsToGo",
        "isHome",
        "preSnapScoreDiff",
        "preSnapTeamWinProbability",
        "quarterMinute",
        "quarterSecond",
        "gameSecondsLeft",
        "quarterSecondsLeft",
        "yardlineNumber",
    ]

    other_columns = [
        col for col in data.columns if col not in y_columns and col not in x_columns
    ]

    return {
        "data": data,
        "columns": {"x": x_columns, "y": y_columns, "other": other_columns},
    }
