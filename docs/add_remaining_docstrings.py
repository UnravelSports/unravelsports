#!/usr/bin/env python
"""
Script to add comprehensive docstrings to remaining high-priority modules.

This script provides the docstrings that should be added to:
1. PressingIntensity (unravel/soccer/models/pressing_intensity.py)
2. EFPI (unravel/soccer/models/formations/efpi.py)
3. BigDataBowlDataset (unravel/american_football/dataset/dataset.py)
4. AmericanFootballGraphConverter (unravel/american_football/graphs/graph_converter.py)

Copy and paste the docstrings below into the appropriate files.
"""

PRESSING_INTENSITY_CLASS_DOCSTRING = '''"""Compute pressing intensity metrics for soccer tracking data.

    Pressing Intensity quantifies the defensive pressure applied to ball carriers
    by measuring spatial coverage, defender proximity, and velocity components of
    defending players around the ball.

    The metric is based on Voronoi cell analysis and defender positioning relative
    to the ball carrier, providing an intuitive measure of defensive pressure that
    correlates with successful pressing actions.

    Mathematical details and validation can be found in:
    `Bekkers (2024): Pressing Intensity <https://arxiv.org/pdf/2501.04712>`_

    Args:
        dataset (KloppyPolarsDataset): Soccer tracking dataset with computed
            velocities and ball ownership information.

    Attributes:
        dataset (pl.DataFrame): The underlying Polars DataFrame with tracking data.
        settings: Dataset settings including pitch dimensions and frame rate.

    Example:
        >>> from kloppy import sportec
        >>> from unravel.soccer import KloppyPolarsDataset, PressingIntensity
        >>> import polars as pl
        >>>
        >>> # Load tracking data
        >>> kloppy_dataset = sportec.load_open_tracking_data(only_alive=True)
        >>> polars_dataset = KloppyPolarsDataset(kloppy_dataset=kloppy_dataset)
        >>>
        >>> # Compute pressing intensity
        >>> model = PressingIntensity(dataset=polars_dataset)
        >>> result = model.fit(
        ...     start_time=pl.duration(minutes=10, seconds=0),
        ...     end_time=pl.duration(minutes=11, seconds=0),
        ...     period_id=1,
        ...     method="teams",
        ...     ball_method="max",
        ...     speed_threshold=2.0
        ... )
        >>>
        >>> # Analyze results
        >>> home_pressing = result['home']
        >>> print(f"Avg pressing intensity: {home_pressing['pressing_intensity'].mean():.2f}")

    Note:
        The dataset must include ball owning team information and player velocities.
        Use :class:`KloppyPolarsDataset` with default settings to ensure all required
        columns are present.

    See Also:
        :doc:`../tutorials/pressing_intensity`: Complete tutorial with visualization.
        :class:`~unravel.soccer.KloppyPolarsDataset`: Load tracking data.
    """'''

PRESSING_INTENSITY_FIT_DOCSTRING = '''"""Compute pressing intensity for a specified time window.

        Analyzes defensive pressure by measuring how closely and quickly defending
        players approach the ball carrier within the specified time range.

        Args:
            start_time (pl.duration, optional): Start of analysis window. Use
                ``pl.duration(minutes=M, seconds=S)`` format. If None, starts from
                beginning of period. Defaults to None.
            end_time (pl.duration, optional): End of analysis window. If None, runs
                until end of period. Defaults to None.
            period_id (int, optional): Which period to analyze (1 for first half,
                2 for second half). If None, analyzes all periods. Defaults to None.
            method (Literal["teams", "full"], optional): Computation method. "teams"
                computes separately for each team, "full" computes aggregate.
                Defaults to "teams".
            ball_method (Literal["max", "mean", "sum"], optional): How to aggregate
                pressing intensity across multiple defenders. "max" takes highest
                pressure from any defender, "mean" averages across all, "sum" totals
                all defenders. Defaults to "max".
            orient (Literal["home_away", "fixed", "attacking_left_to_right"], optional):
                Coordinate orientation. "home_away" normalizes so home attacks right,
                "fixed" uses original coordinates, "attacking_left_to_right" normalizes
                to attacking direction. Defaults to "home_away".
            speed_threshold (float, optional): Minimum speed (m/s) for a player to be
                considered actively pressing. Players moving slower are excluded.
                Defaults to 2.0.

        Returns:
            Union[Dict[str, pl.DataFrame], pl.DataFrame]: If method="teams", returns
                dict with 'home' and 'away' DataFrames. If method="full", returns
                single DataFrame. Each DataFrame contains:

                - frame_id: Frame identifier
                - timestamp: Time within period
                - pressing_intensity: Computed metric value
                - ball_x, ball_y: Ball position
                - (additional metadata columns)

        Raises:
            ValueError: If start_time >= end_time.
            ValueError: If period_id is invalid.
            KeyError: If required columns are missing from dataset.

        Example:
            >>> # Analyze first 5 minutes of each half
            >>> result_h1 = model.fit(
            ...     start_time=pl.duration(minutes=0),
            ...     end_time=pl.duration(minutes=5),
            ...     period_id=1,
            ...     method="teams"
            ... )
            >>>
            >>> # Find peak pressing moment
            >>> home_max = result_h1['home'].filter(
            ...     pl.col('pressing_intensity') == pl.col('pressing_intensity').max()
            ... )
            >>> print(f"Peak pressing at {home_max['timestamp'].item()}")

        Note:
            Higher values indicate more intense pressing. Typical values range from
            0 (no pressure) to ~10 (extreme pressure). Values above 5 usually indicate
            successful pressing situations.

        Warning:
            If ball owning team was inferred (not provided in original data), pressing
            intensity may be inaccurate during contested ball situations.
    """'''

EFPI_CLASS_DOCSTRING = '''"""Detect team formations and assign player positions using template matching.

    EFPI (Elastic Formation and Position Identification) uses template matching with
    linear assignment to identify which formation (e.g., 4-4-2, 4-3-3) a team is using
    and assign each player to a tactical position (e.g., CB, CM, LW).

    The algorithm:
    1. Compares player positions to 65 pre-defined formation templates
    2. Uses elastic matching to handle positional variations
    3. Applies Hungarian algorithm for optimal player-position assignment
    4. Tracks formation changes over time with configurable stability

    Mathematical formulation and evaluation in:
    `Bekkers (2025): EFPI <https://arxiv.org/pdf/2506.23843>`_

    Args:
        dataset (KloppyPolarsDataset): Soccer tracking dataset with player positions.

    Attributes:
        dataset (pl.DataFrame): The underlying Polars DataFrame.
        formations: Available formation templates (65 by default).

    Example:
        >>> from kloppy import sportec
        >>> from unravel.soccer import KloppyPolarsDataset, EFPI
        >>>
        >>> # Load tracking data
        >>> kloppy_dataset = sportec.load_open_tracking_data(only_alive=True)
        >>> polars_dataset = KloppyPolarsDataset(kloppy_dataset=kloppy_dataset)
        >>>
        >>> # Detect formations
        >>> model = EFPI(dataset=polars_dataset)
        >>> result = model.fit(
        ...     formations=None,  # Use all 65 formations
        ...     every="5m",  # Detect every 5 minutes
        ...     substitutions="drop",
        ...     change_threshold=0.1
        ... )
        >>>
        >>> # Analyze detected formations
        >>> for detection in result.detected_formations:
        ...     print(f"{detection.formation_name} from {detection.start_time} "
        ...           f"to {detection.end_time}")

    Note:
        Requires at least 8-10 outfield players for reliable detection. Goalkeepers
        are automatically identified.

    Warning:
        Formations are detected from average positions, so rapid tactical adjustments
        (e.g., during corners) may not be captured accurately.

    See Also:
        :doc:`../tutorials/formation_detection`: Complete tutorial.
        :class:`~unravel.soccer.KloppyPolarsDataset`: Load tracking data.
    """'''

EFPI_FIT_DOCSTRING = '''"""Detect formations and assign positions for the dataset.

        Analyzes player positions over specified time intervals to identify formations
        and tactical positions, tracking changes throughout the match.

        Args:
            formations (List[str], optional): Formation codes to consider (e.g.,
                ["442", "433", "4231"]). If None, uses all 65 default formations.
                Defaults to None.
            every (str, optional): Time granularity for detection. Options:
                - "frame": Every single frame (most detailed, slowest)
                - "1m", "5m", etc.: Time intervals (e.g., every 1 minute)
                - "possession": Once per possession
                - "period": Once per period
                Uses Polars duration syntax. Defaults to "5m".
            substitutions (Literal["drop", "keep", "interpolate"], optional):
                How to handle substitution frames. "drop" removes them (cleanest),
                "keep" includes them, "interpolate" fills across them.
                Defaults to "drop".
            change_threshold (float, optional): Minimum difference (0-1) required
                to register a formation change. Higher values = less sensitive.
                Defaults to 0.1.
            change_after_possession (bool, optional): If True, only allow formation
                changes at possession boundaries. More realistic than allowing
                mid-possession changes. Defaults to True.

        Returns:
            DetectedFormations: Object containing:
                - detected_formations: List of DetectedFormation objects
                - Each DetectedFormation has:
                    - formation_name: Formation code (e.g., "442")
                    - team_id: Which team
                    - period_id: Which period
                    - start_time, end_time: Time range
                    - player_positions: Dict[player_id -> position_name]
                    - confidence: Match quality score

        Raises:
            ValueError: If formations list contains invalid codes.
            ValueError: If every parameter has invalid format.

        Example:
            >>> # Detect common formations only, every minute
            >>> result = model.fit(
            ...     formations=["442", "433", "4231", "352"],
            ...     every="1m",
            ...     substitutions="drop",
            ...     change_threshold=0.15  # Less sensitive
            ... )
            >>>
            >>> # Count formation usage
            >>> from collections import Counter
            >>> formation_counts = Counter(
            ...     d.formation_name for d in result.detected_formations
            ... )
            >>> print(f"Most common: {formation_counts.most_common(1)[0]}")
            >>>
            >>> # Find when formation changed
            >>> for i in range(1, len(result.detected_formations)):
            ...     if result.detected_formations[i].formation_name != \\
            ...        result.detected_formations[i-1].formation_name:
            ...         print(f"Changed at {result.detected_formations[i].start_time}")

        Note:
            Detection quality improves with longer time windows. Frame-by-frame
            detection may be noisy; use "1m" or longer for stable results.

        Warning:
            Template-based matching works best for traditional formations. Highly
            fluid or asymmetric tactical setups may not match templates well.
    """'''

BIG_DATA_BOWL_CLASS_DOCSTRING = '''"""Load and process NFL Big Data Bowl tracking data.

    This class loads NFL tracking data from the Big Data Bowl competition CSV files,
    merging player metadata and play information into a unified Polars DataFrame.

    The Big Data Bowl data includes:
    - Player tracking (position, speed, acceleration, orientation)
    - Player metadata (height, weight, position)
    - Play information (down, distance, formation, personnel)

    Args:
        tracking_file_path (str): Path to tracking CSV file (e.g., "tracking_week_1.csv").
        players_file_path (str): Path to players CSV file with player metadata.
        plays_file_path (str): Path to plays CSV file with play-level information.
        **kwargs: Additional keyword arguments passed to DefaultDataset.

    Attributes:
        data (pl.DataFrame): Merged DataFrame with tracking, player, and play data.
        settings (DefaultSettings): Configuration and metadata.

    Example:
        >>> from unravel.american_football import BigDataBowlDataset
        >>>
        >>> # Load Big Data Bowl data
        >>> dataset = BigDataBowlDataset(
        ...     tracking_file_path="tracking_week_1.csv",
        ...     players_file_path="players.csv",
        ...     plays_file_path="plays.csv"
        ... )
        >>>
        >>> # Access data
        >>> df = dataset.data
        >>> print(df.head())
        >>>
        >>> # Add labels and graph IDs for GNN training
        >>> dataset.add_dummy_labels(by=["gameId", "playId"])
        >>> dataset.add_graph_ids(by=["gameId", "playId"])

    Note:
        Big Data Bowl data is available from Kaggle competitions. Download the
        CSV files from the competition page.

    See Also:
        :doc:`../tutorials/american_football`: Complete tutorial.
        :class:`~unravel.american_football.AmericanFootballGraphConverter`: Convert to graphs.
        `Big Data Bowl <https://www.kaggle.com/c/nfl-big-data-bowl-2025>`_: Competition page.
    """'''

AMERICAN_FOOTBALL_CONVERTER_CLASS_DOCSTRING = '''"""Convert NFL tracking data to graph structures for GNN training.

    Similar to :class:`~unravel.soccer.SoccerGraphConverter` but adapted for American
    Football with sport-specific features:
    - 20 node features (vs 12 for soccer) including player size, orientation
    - 9 edge features including relative orientation and combined metrics
    - Line of scrimmage and goal line distances
    - Personnel groupings and formations

    The converter supports PyTorch Geometric (recommended) and Spektral (deprecated).

    Args:
        dataset (BigDataBowlDataset): Big Data Bowl dataset with tracking data.
        chunk_size (int, optional): Number of graphs to process simultaneously.
            Defaults to 20000.
        **kwargs: Additional parameters inherited from DefaultGraphConverter.

    Attributes:
        settings (GraphSettingsPolars): Graph configuration settings.
        n_node_features (int): Total node features (20 default).
        n_edge_features (int): Total edge features (9 default).
        n_graph_features (int): Total global features.

    Example:
        >>> from unravel.american_football import BigDataBowlDataset, AmericanFootballGraphConverter
        >>>
        >>> # Load data
        >>> dataset = BigDataBowlDataset(
        ...     tracking_file_path="tracking_week_1.csv",
        ...     players_file_path="players.csv",
        ...     plays_file_path="plays.csv"
        ... )
        >>> dataset.add_dummy_labels(by=["gameId", "playId"])
        >>> dataset.add_graph_ids(by=["gameId", "playId"])
        >>>
        >>> # Convert to graphs
        >>> converter = AmericanFootballGraphConverter(
        ...     dataset=dataset,
        ...     self_loop_ball=True,
        ...     adjacency_matrix_type="split_by_team"
        ... )
        >>> graphs = converter.to_pytorch_graphs()

    Note:
        American Football graphs typically have variable numbers of players on field
        (especially for special teams plays). Consider using ``pad=True`` or handling
        variable graph sizes in your model.

    See Also:
        :class:`~unravel.soccer.SoccerGraphConverter`: Similar converter for soccer.
        :doc:`../tutorials/american_football`: Complete tutorial.
    """'''

# Print instructions
print("="*80)
print("DOCSTRINGS TO ADD")
print("="*80)
print()
print("1. unravel/soccer/models/pressing_intensity.py")
print("   Replace class docstring:")
print(PRESSING_INTENSITY_CLASS_DOCSTRING)
print()
print("   Replace fit() method docstring:")
print(PRESSING_INTENSITY_FIT_DOCSTRING)
print()
print("-"*80)
print()
print("2. unravel/soccer/models/formations/efpi.py")
print("   Replace class docstring:")
print(EFPI_CLASS_DOCSTRING)
print()
print("   Replace fit() method docstring:")
print(EFPI_FIT_DOCSTRING)
print()
print("-"*80)
print()
print("3. unravel/american_football/dataset/dataset.py")
print("   Replace class docstring:")
print(BIG_DATA_BOWL_CLASS_DOCSTRING)
print()
print("-"*80)
print()
print("4. unravel/american_football/graphs/graph_converter.py")
print("   Replace class docstring:")
print(AMERICAN_FOOTBALL_CONVERTER_CLASS_DOCSTRING)
print()
print("="*80)
print()
print("After adding these docstrings, rebuild the documentation:")
print("  cd docs")
print("  make clean && make html")
print("  open build/html/index.html")
print()
print("="*80)
