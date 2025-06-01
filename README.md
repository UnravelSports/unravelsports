![unravelsports logo](https://github.com/UnravelSports/unravelsports.github.io/blob/main/imgs/unravelsports-5500x800.png?raw=true)
<div align="right">

[![Powered by PySport](https://img.shields.io/badge/powered%20by-PySport-orange.svg?style=flat&colorA=C3C3C3&colorB=4B99CC)](https://pysport.org) 
[![Powered by PySport](https://img.shields.io/badge/powered%20by-UnravelSports-orange.svg?style=flat&colorB=E6B611&colorA=C3C3C3)](https://unravelsports.github.io/)
[![tl;dr legal](https://img.shields.io/badge/license-Mozilla%20Public%20License%20v2.0-orange.svg?style=flat&colorA=C3C3C3&colorB=E20E6A)](https://www.tldrlegal.com/license/mozilla-public-license-2-0-mpl-2) 
</div>

🌀 `pip install unravelsports`


🌀 What is it?
-----

The **unravelsports** package aims to aid researchers, analysts and enthusiasts by providing intermediary steps in the complex process of converting raw sports data into meaningful information and actionable insights.

This package currently supports:
- ⚽ 🏈 [**Polars DataFrame Conversion**](#polars-dataframes) 
- ⚽ 🏈 [**Graph Neural Network**](#graph-neural-networks) Training, Graph Conversion and Prediction <small>
  [[Bekkers & Sahasrabudhe (2023)](https://arxiv.org/pdf/2411.17450)]</small>
- ⚽ [**Pressing Intensity**](#pressing-intensity) 
  <small>[[Bekkers (2024)](https://arxiv.org/pdf/2501.04712)]</small>

🌀 Features
-----

### **Polars DataFrames**

⚽🏈 **Convert Tracking Data** into [Polars DataFrames](https://pola.rs/) for rapid data conversion and data processing. 

⚽ For soccer we rely on [Kloppy](https://kloppy.pysport.org/) and as such we support _Sportec_$^1$, _SkillCorner_$^1$, _PFF_$^{1, 2}$, _Metrica_$^1$, _StatsPerform_, _Tracab (CyronHego)_ and _SecondSpectrum_ tracking data.
```python
from unravel.soccer import KloppyPolarsDataset

from kloppy import sportec

kloppy_dataset = sportec.load_open_tracking_data()
kloppy_polars_dataset = KloppyPolarsDataset(
    kloppy_dataset=kloppy_dataset
)
```
|    |   period_id | timestamp       |   frame_id | ball_state   | id             |      x |     y |   z | team_id        | position_name   | game_id        |     vx |     vy |   vz |     v |   ax |   ay |   az |   a | ball_owning_team_id   | is_ball_carrier   |
|---:|------------:|:----------------|-----------:|:-------------|:---------------|-------:|------:|----:|:---------------|:----------------|:---------------|-------:|-------:|-----:|------:|-----:|-----:|-----:|----:|:----------------------|:------------------|
|  0 |           1 | 0 days 00:00:00 |      10000 | alive        | DFL-OBJ-00008F | -20.67 | -4.56 |   0 | DFL-CLU-000005 | RCB             | DFL-MAT-J03WPY |  0.393 | -0.214 |    0 | 0.447 |    0 |    0 |    0 |   0 | DFL-CLU-00000P        | False             |
|  1 |           1 | 0 days 00:00:00 |      10000 | alive        | DFL-OBJ-0000EJ |  -8.86 | -0.94 |   0 | DFL-CLU-000005 | UNK             | DFL-MAT-J03WPY | -0.009 |  0.018 |    0 | 0.02  |    0 |    0 |    0 |   0 | DFL-CLU-00000P        | False             |
|  2 |           1 | 0 days 00:00:00 |      10000 | alive        | DFL-OBJ-0000F8 |  -2.12 |  9.85 |   0 | DFL-CLU-00000P | RM              | DFL-MAT-J03WPY |  0     |  0     |    0 | 0     |    0 |    0 |    0 |   0 | DFL-CLU-00000P        | False             |
|  3 |           1 | 0 days 00:00:00 |      10000 | alive        | DFL-OBJ-0000NZ |   0.57 | 23.23 |   0 | DFL-CLU-00000P | RB              | DFL-MAT-J03WPY |  0.179 | -0.134 |    0 | 0.223 |    0 |    0 |    0 |   0 | DFL-CLU-00000P        | False             |
|  4 |           1 | 0 days 00:00:00 |      10000 | alive        | DFL-OBJ-0001HW | -46.26 |  0.08 |   0 | DFL-CLU-000005 | GK              | DFL-MAT-J03WPY |  0.357 |  0.071 |    0 | 0.364 |    0 |    0 |    0 |   0 | DFL-CLU-00000P        | False             |


$^1$ <small>Open data available through kloppy.</small>

$^2$ <small>Currently unreleased in kloppy, only available through kloppy master branch. [Click here for World Cup 2022 Dataset](https://www.blog.fc.pff.com/blog/enhanced-2022-world-cup-dataset)</small> 

🏈 For American Football we use [BigDataBowl Data](https://www.kaggle.com/competitions/nfl-big-data-bowl-2025/data) directly.

```python
from unravel.american_football import BigDataBowlDataset

bdb = BigDataBowlDataset(
    tracking_file_path="week1.csv",
    players_file_path="players.csv",
    plays_file_path="plays.csv",
)
```

### **Graph Neural Networks**

⚽🏈 Convert **[Polars Dataframes](#polars-dataframes)** into [Graphs](examples/graphs_faq.md) to train **graph neural networks**. These [Graphs](examples/graphs_faq.md) can be used with [**Spektral**](https://github.com/danielegrattarola/spektral) - a flexible framework for training graph neural networks. 
`unravelsports` allows you to **randomize** and **split** data into train, test and validation sets along matches, sequences or possessions to avoid leakage and improve model quality. And finally, **train**, **validate** and **test** your (custom) Graph model(s) and easily **predict** on new data.

```python
converter = SoccerGraphConverterPolars(
    dataset=kloppy_polars_dataset,
    self_loop_ball=True,
    adjacency_matrix_connect_type="ball",
    adjacency_matrix_type="split_by_team",
    label_type="binary",
    defending_team_node_value=0.1,
    non_potential_receiver_node_value=0.1,
    random_seed=False,
    pad=False,
    verbose=False,
)
```

### **Pressing Intensity**

Compute [**Pressing Intensity**](https://arxiv.org/abs/2501.04712) for a whole game (or segment) of Soccer tracking data.

See [**Pressing Intensity Jupyter Notebook**](examples/pressing_intensity.ipynb) for an example how to create mp4 videos.

```python
from unravel.soccer import PressingIntensity

import polars as pl

model = PressingIntensity(
    dataset=kloppy_polars_dataset
)
model.fit(
    start_time = pl.duration(minutes=1, seconds=53),
    end_time = pl.duration(minutes=2, seconds=32),
    period_id = 1,
    method="teams",
    ball_method="max",
    orient="home_away",
    speed_threshold=2.0,
) 
```

![1. FC Köln vs. FC Bayern München (May 27th 2023)](assets/gif/preview.gif)

⌛ ***More to come soon...!***

🌀 Quick Start
-----
📖 ⚽ The [**Quick Start Jupyter Notebook**](examples/0_quick_start_guide.ipynb) explains how to convert any positional tracking data from **Kloppy** to **Spektral GNN** in a few easy steps while walking you through the most important features and documentation.

📖 ⚽ The [**Graph Converter Tutorial Jupyter Notebook**](examples/1_kloppy_gnn_train.ipynb) gives an in-depth walkthrough.

📖 🏈 The [**BigDataBowl Converter Tutorial Jupyter Notebook**](examples/2_big_data_bowl_guide.ipynb) gives an guide on how to convert the BigDataBowl data into Graphs.

📖 ⚽ The [**Pressing Intensity Tutorial Jupyter Notebook**](examples/pressing_intensity.ipynb) gives a description on how to create Pressing Intensity videos.


🌀 Documentation
-----
For now, follow the [**Graph Converter Tutorial**](examples/1_kloppy_gnn_train.ipynb) and check the [**Graph FAQ**](examples/graphs_faq.md), more documentation will follow!

Additional reading:

📖 [A Graph Neural Network Deep-dive into Successful Counterattacks {A. Sahasrabudhe & J. Bekkers, 2023}](https://github.com/USSoccerFederation/ussf_ssac_23_soccer_gnn/tree/main)

🌀 Installation
----
The easiest way to get started is:

```bash
pip install unravelsports
```

⚠️ Due to compatibility issues **unravelsports** currently only works on Python 3.11 with:
```
spektral==1.20.0 
tensorflow==2.14.0 
keras==2.14.0
kloppy==3.16.0
polars==1.2.1
```
These dependencies come pre-installed with the package. It is advised to create a [virtual environment](https://virtualenv.pypa.io/en/latest/).

This package is tested on the latest versions of Ubuntu, MacOS and Windows. 

🌀 Licenses
----
This project is licensed under the [Mozilla Public License Version 2.0 (MPL)](LICENSE), which requires that you include a copy of the license and provide attribution to the original authors. Any modifications you make to the MPL-licensed files must be documented, and the source code for those modifications must be made open-source under the same license.

🌀 Citation
----
If you use this repository for any educational purposes, research, project etc., please reference both:

📎 [The `unravelsports` package](https://github.com/unravelsports/unravelsports).
<details>
<summary>BibTex</summary>
<pre>
@software{unravelsports2024repository,
  author = {Bekkers, Joris},
  title = {unravelsports},
  version = {0.3.0},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/unravelsports/unravelsports}
}
</pre>
</details>

<br>

📎 [Bekkers, J., & Sahasrabudhe, A. (2024). A Graph Neural Network deep-dive into successful counterattacks. arXiv preprint arXiv:2411.17450.](https://arxiv.org/pdf/2411.17450)
<details>
<summary>BibTex</summary>
<pre>
@inproceedings{sahasrabudhe2023graph,
  title={A Graph Neural Network deep-dive into successful counterattacks},
  author={Sahasrabudhe, Amod and Bekkers, Joris},
  booktitle={17th Annual MIT Sloan Sports Analytics Conference. Boston, MA, USA: MIT},
  pages={15},
  year={2023}
}
</pre>
</details>
<br>

📎 [Bekkers, J. (2024). Pressing Intensity: An Intuitive Measure for Pressing in Soccer. arXiv preprint arXiv:2501.04712.](https://arxiv.org/pdf/2501.04712)
<details>
<summary>BibTex</summary>
<pre>
@article{bekkers2024pressing,
  title={Pressing Intensity: An Intuitive Measure for Pressing in Soccer},
  author={Bekkers, Joris},
  journal={arXiv preprint arXiv:2501.04712},
  year={2024}
}
</pre>
</details>
<br>

🌀 Social Media
----
[<img alt="alt_text" width="40px" src="https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png"/>](https://www.linkedin.com/in/joris-bekkers-33138288/)
[<img alt="alt_text" width="40px" src="https://upload.wikimedia.org/wikipedia/commons/thumb/7/7a/Bluesky_Logo.svg/2319px-Bluesky_Logo.svg.png"/>](https://bsky.app/profile/unravelsports.com)



