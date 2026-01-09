#!/usr/bin/env python
"""
Script to generate API documentation RST files for unravelsports.
Run this from the docs/ directory.
"""

import os

# Create directories
os.makedirs("source/api/soccer", exist_ok=True)
os.makedirs("source/api/american_football", exist_ok=True)
os.makedirs("source/api/utils", exist_ok=True)

# Soccer API index
soccer_index = """Soccer
======

Soccer-specific functionality for tracking data analysis.

.. currentmodule:: unravel.soccer

The soccer module provides tools for loading, processing, and analyzing soccer tracking data,
including dataset conversion, graph creation, and tactical models.

.. toctree::
   :maxdepth: 2

   soccer/dataset
   soccer/graphs
   soccer/models

* :class:`~unravel.soccer.KloppyPolarsDataset` - Convert Kloppy data to Polars
* :class:`~unravel.soccer.SoccerGraphConverter` - Convert tracking data to graphs
* :class:`~unravel.soccer.PressingIntensity` - Compute pressing intensity
* :class:`~unravel.soccer.EFPI` - Formation and position detection
"""

# Soccer dataset
soccer_dataset = """Dataset
=======

Loading and converting soccer tracking data.

.. currentmodule:: unravel.soccer

.. autoclass:: KloppyPolarsDataset
   :members:
   :undoc-members:
   :show-inheritance:

.. code-block:: python

   from kloppy import sportec
   from unravel.soccer import KloppyPolarsDataset

   # Load tracking data
   kloppy_dataset = sportec.load_open_tracking_data(only_alive=True)

   # Convert to Polars
   polars_dataset = KloppyPolarsDataset(kloppy_dataset=kloppy_dataset)

   # Access the DataFrame
   df = polars_dataset.dataset
   print(df.head())
"""

# Soccer graphs
soccer_graphs = """Graphs
======

Converting soccer tracking data to graph structures for GNN training.

.. currentmodule:: unravel.soccer

.. autoclass:: SoccerGraphConverter
   :members:
   :undoc-members:
   :show-inheritance:

.. code-block:: python

   from unravel.soccer import SoccerGraphConverter

   converter = SoccerGraphConverter(
       dataset=polars_dataset,
       self_loop_ball=True,
       adjacency_matrix_connect_type="ball",
       adjacency_matrix_type="split_by_team",
       label_type="binary",
   )

   # Convert to PyTorch Geometric
   graphs = converter.to_pytorch_graphs()

   # Or Spektral (deprecated)
   graphs = converter.to_spektral_graphs()
"""

# Soccer models
soccer_models = """Models
======

Soccer-specific analytical models.

.. currentmodule:: unravel.soccer

.. autoclass:: PressingIntensity
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: EFPI
   :members:
   :undoc-members:
   :show-inheritance:

.. code-block:: python

   from unravel.soccer import PressingIntensity
   import polars as pl

   model = PressingIntensity(dataset=polars_dataset)
   result = model.fit(
       start_time=pl.duration(minutes=1, seconds=53),
       end_time=pl.duration(minutes=2, seconds=32),
       period_id=1,
       method="teams",
   )

.. code-block:: python

   from unravel.soccer import EFPI

   model = EFPI(dataset=polars_dataset)
   formations = model.fit(
       every="5m",
       substitutions="drop",
   )
"""

# American Football index
af_index = """American Football
=================

American Football (NFL) specific functionality.

.. currentmodule:: unravel.american_football

The american_football module provides tools for loading and analyzing NFL tracking data from
the Big Data Bowl.

.. toctree::
   :maxdepth: 2

   american_football/dataset
   american_football/graphs

* :class:`~unravel.american_football.BigDataBowlDataset` - Load NFL tracking data
* :class:`~unravel.american_football.AmericanFootballGraphConverter` - Convert to graphs
"""

# American Football dataset
af_dataset = """Dataset
=======

Loading NFL tracking data from Big Data Bowl.

.. currentmodule:: unravel.american_football

.. autoclass:: BigDataBowlDataset
   :members:
   :undoc-members:
   :show-inheritance:

.. code-block:: python

   from unravel.american_football import BigDataBowlDataset

   bdb_dataset = BigDataBowlDataset(
       tracking_file_path="tracking_week_1.csv",
       players_file_path="players.csv",
       plays_file_path="plays.csv",
   )

   # Access the DataFrame
   df = bdb_dataset.dataset
   print(df.head())

"""

# American Football graphs
af_graphs = """Graphs
======

Converting NFL tracking data to graph structures.

.. currentmodule:: unravel.american_football

.. autoclass:: AmericanFootballGraphConverter
   :members:
   :undoc-members:
   :show-inheritance:

.. code-block:: python

   from unravel.american_football import AmericanFootballGraphConverter

   converter = AmericanFootballGraphConverter(
       dataset=bdb_dataset,
       self_loop_ball=True,
       adjacency_matrix_connect_type="ball",
       adjacency_matrix_type="split_by_team",
       label_type="binary",
   )

   graphs = converter.to_pytorch_graphs()

"""

# Utils index
utils_index = """Utils
=====

Utility functions and base classes.

.. currentmodule:: unravel.utils

The utils module provides helper functions, base classes, and utilities used throughout
the unravelsports package.

.. toctree::
   :maxdepth: 2

   utils/objects
   utils/features
   utils/helpers

.. autosummary::
   :toctree: generated/
   :nosignatures:

   dummy_labels
   dummy_graph_ids
   add_dummy_label_column
   add_graph_id_column
"""

# Utils objects
utils_objects = """Objects
=======

Base classes and core data structures.

.. currentmodule:: unravel.utils

.. autoclass:: GraphDataset
   :members:
   :undoc-members:
   :show-inheritance:

.. code-block:: python

   from unravel.utils import GraphDataset

   # Create dataset
   dataset = GraphDataset(graphs=graphs, format="pyg")

   # Split data
   train, test, val = dataset.split_test_train_validation(4, 1, 1)
"""

# Utils features
utils_features = """Features
========

Feature engineering utilities and decorators.

.. currentmodule:: unravel.utils.features

The features module provides built-in feature functions and a decorator for creating custom
graph features.

.. autofunction:: graph_feature

.. autofunction:: x_normed
.. autofunction:: y_normed
.. autofunction:: speeds_normed
.. autofunction:: velocity_components_2d_normed

.. autofunction:: distance_normed
.. autofunction:: angle_features

.. code-block:: python

   from unravel.utils.features import graph_feature

   @graph_feature(
       cols=["x", "y"],
       returns=["distance_from_center"],
       type="node"
   )
   def distance_from_center(x, y):
       import polars as pl
       return [pl.sqrt(x**2 + y**2)]

   # Use in converter
   from unravel.soccer import SoccerGraphConverter

   converter = SoccerGraphConverter(
       dataset=polars_dataset,
       node_feature_cols=[distance_from_center],
   )
"""

# Utils helpers
utils_helpers = """Helpers
=======

Helper functions for data manipulation.

.. currentmodule:: unravel.utils

.. autofunction:: dummy_labels
.. autofunction:: add_dummy_label_column

.. autofunction:: dummy_graph_ids
.. autofunction:: add_graph_id_column

.. code-block:: python

   from unravel.utils import add_dummy_label_column

   # Add random binary labels
   dataset.dataset = add_dummy_label_column(dataset.dataset)

Adding Graph IDs
~~~~~~~~~~~~~~~~

.. code-block:: python

   from unravel.utils import add_graph_id_column

   # Each frame is a separate graph
   dataset.dataset = add_graph_id_column(dataset.dataset, by=["frame_id"])

   # Group by possession
   dataset.dataset = add_graph_id_column(
       dataset.dataset,
       by=["ball_owning_team_id", "period_id"]
   )
"""

# Additional information files
citations = """Citations
=========

If you use unravelsports in your research or project, please cite the relevant papers:

The Package
-----------

.. code-block:: bibtex

   @software{unravelsports2024repository,
     author = {Bekkers, Joris},
     title = {unravelsports},
     version = {2.0.0},
     year = {2024},
     publisher = {GitHub},
     url = {https://github.com/unravelsports/unravelsports}
   }

Graph Neural Networks
---------------------

.. code-block:: bibtex

   @inproceedings{sahasrabudhe2023graph,
     title={A Graph Neural Network deep-dive into successful counterattacks},
     author={Sahasrabudhe, Amod and Bekkers, Joris},
     booktitle={17th Annual MIT Sloan Sports Analytics Conference},
     year={2023}
   }

Bekkers, J., & Sahasrabudhe, A. (2024). A Graph Neural Network deep-dive into successful
counterattacks. `arXiv:2411.17450 <https://arxiv.org/pdf/2411.17450>`_

Pressing Intensity
------------------

.. code-block:: bibtex

   @article{bekkers2024pressing,
     title={Pressing Intensity: An Intuitive Measure for Pressing in Soccer},
     author={Bekkers, Joris},
     journal={arXiv preprint arXiv:2501.04712},
     year={2024}
   }

Bekkers, J. (2024). Pressing Intensity: An Intuitive Measure for Pressing in Soccer.
`arXiv:2501.04712 <https://arxiv.org/pdf/2501.04712>`_

Formation Detection (EFPI)
--------------------------

.. code-block:: bibtex

   @article{bekkers2025efpi,
     title={EFPI: Elastic Formation and Position Identification in Football (Soccer)
            using Template Matching and Linear Assignment},
     author={Bekkers, Joris},
     journal={arXiv preprint arXiv:2506.23843},
     year={2025}
   }

Bekkers, J. (2025). EFPI: Elastic Formation and Position Identification in Football (Soccer)
using Template Matching and Linear Assignment. `arXiv:2506.23843 <https://arxiv.org/pdf/2506.23843>`_
"""

license_page = """License
=======

unravelsports is licensed under the Mozilla Public License Version 2.0 (MPL 2.0).

Summary
-------

The MPL 2.0 is a copyleft license that:

* **Allows**: Commercial use, modification, distribution, and private use
* **Requires**: Source code disclosure for modifications, license and copyright notice, same license for modifications
* **Permits**: Use in proprietary software (as long as MPL-licensed files remain open)

Full License Text
-----------------

The complete license text can be found in the `LICENSE file <https://github.com/unravelsports/unravelsports/blob/main/LICENSE>`_
in the repository.

For a human-readable summary, see the `tl;dr Legal page for MPL 2.0 <https://www.tldrlegal.com/license/mozilla-public-license-2-0-mpl-2>`_.

Key Requirements
----------------

When using unravelsports in your project:

1. **Include License**: Include a copy of the MPL 2.0 license
2. **Attribute**: Provide attribution to the original authors
3. **Document Changes**: If you modify MPL-licensed files, document the changes
4. **Share Modifications**: Make your modifications to MPL-licensed files available under MPL 2.0

Using in Closed-Source Projects
--------------------------------

You **can** use unravelsports in closed-source projects, but:

* Any modifications to unravelsports source files must be released under MPL 2.0
* You can combine it with proprietary code (as a library/dependency)
* Your proprietary code does not need to be open-sourced

Questions?
----------

For questions about licensing, please open an issue on
`GitHub <https://github.com/unravelsports/unravelsports/issues>`_.
"""


# Write all files
files_to_write = {
    "source/api/soccer.rst": soccer_index,
    "source/api/soccer/dataset.rst": soccer_dataset,
    "source/api/soccer/graphs.rst": soccer_graphs,
    "source/api/soccer/models.rst": soccer_models,
    "source/api/american_football.rst": af_index,
    "source/api/american_football/dataset.rst": af_dataset,
    "source/api/american_football/graphs.rst": af_graphs,
    "source/api/utils.rst": utils_index,
    "source/api/utils/objects.rst": utils_objects,
    "source/api/utils/features.rst": utils_features,
    "source/api/utils/helpers.rst": utils_helpers,
    "source/additional/citations.rst": citations,
    "source/additional/license.rst": license_page,
}

for filepath, content in files_to_write.items():
    full_path = os.path.join(os.path.dirname(__file__), filepath)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, "w") as f:
        f.write(content)
    print(f"Created: {filepath}")

print("\nAll API documentation files generated successfully!")
print("Run 'make html' from the docs/ directory to build the documentation.")
