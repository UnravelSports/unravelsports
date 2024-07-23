![unravelsports logo](https://github.com/UnravelSports/unravelsports.github.io/blob/main/imgs/unravelsports-5500x800-4.png?raw=true)
<div align="right">

[![Powered by PySport](https://img.shields.io/badge/powered%20by-PySport-orange.svg?style=flat&colorA=104467&colorB=007D8A)](https://pysport.org)
</div>

🌀 `pip install unravelsports`


🌀 What is it?
-----

The **unravelsports** package aims to aid researchers, analysts and enthusiasts by providing intermediary steps in the complex process of turning raw sports data into meaningful information and actionable insights.

🌀 Features
-----

⚽ Convert **positional soccer data** into graphs to train **graph neural networks** by leveraging the powerful [**Kloppy**](https://github.com/PySport/kloppy/tree/master) data conversion standard and [**Spektral**](https://github.com/danielegrattarola/spektral) - a flexible framework for training graph neural networks. 

⚽ Randomize and split data into **train, test and validation sets** along matches, sequences or possessions to avoid leakage and improve model quality.

⚽ Train, validate and test your (custom) Graph model(s) and easily predict on new data.

⚽ Leverage the power of **Kloppy** standardization and use **unravelsports** to execute the above for _Metrica_, _Sportec_, _Tracab (CyronHego)_, _SecondSpectrum_, _SkillCorner_ and _StatsPerform_ tracking data.

⌛ ***More to come soon...!***

🌀 Getting Started
-----
📖 The [**Getting Started Jupyter Notebook**](examples/0_quick_start_guide.ipynb) explains how to convert any positional tracking data from **Kloppy** to **Spektral GNN** in a few easy steps while walking you through the most important features and documentation.

📖 The [**Graph Converter Tutorial Jupyter Notebook**](examples/1_kloppy_gnn_train.ipynb) gives an in-depth walkthrough.

🌀 Documentation
-----
For now, follow the [**Graph Converter Tutorial**](examples/1_kloppy_gnn_train.ipynb), more documentation will follow!

Additional reading:

📖 [A Graph Neural Network Deep-dive into Successful Counterattacks {A. Sahasrabudhe & J. Bekkers, 2023}](https://github.com/USSoccerFederation/ussf_ssac_23_soccer_gnn/tree/main)

🌀 Installation
----
Due to compatibility issue **unravelsports** currently only works on Python 3.11 with spektral==1.20.0 tensorflow==2.14.0 and keras==2.14.0. It is tested on the latest versions of Ubuntu, MacOS and Windows.

The easiest way to get started is by running:

```bash
pip install unravelsports
```

🌀 Contributing
----
All contributions, bug reports, bug fixes, documentation improvements, enhancements, and ideas are welcome. Please don't hesitate to create a Pull Request for any non-game winning features you have implemented!

An overview on how to contribute can be found in the [**contributing guide**](CONTRIBUTING.md).

🌀 Citation
----
If you use this repository for any educational purposes, research, project etc., please reference either/or:

📎 [Bekkers, J., & Dabadghao, S. (2019). Flow motifs in soccer: What can passing behavior tell us?. Journal of Sports Analytics, 5(4), 299-311.](https://content.iospress.com/download/journal-of-sports-analytics/jsa190290?id=journal-of-sports-analytics%2Fjsa190290)
<details>
<summary>BibTex</summary>
<pre>
@article{bekkers2019flow,
  title={Flow motifs in soccer: What can passing behavior tell us?},
  author={Bekkers, Joris and Dabadghao, Shaunak},
  journal={Journal of Sports Analytics},
  volume={5},
  number={4},
  pages={299--311},
  year={2019},
  publisher={IOS Press}
}
</pre>
</details>

<br>

📎 [Sahasrabudhe, A., & Bekkers, J. (2023). A graph neural network deep-dive into successful counterattacks. MIT Sloan Sports Analytics Conference.](https://ussf-ssac-23-soccer-gnn.s3.us-east-2.amazonaws.com/public/Sahasrabudhe_Bekkers_SSAC23.pdf)
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
and let me know on

[<img alt="alt_text" width="40px" src="https://github.com/USSoccerFederation/ussf_ssac_23_soccer_gnn/blob/main/img/linkedin.png?raw=true"/>](https://www.linkedin.com/in/joris-bekkers-33138288/)
[<img alt="alt_text" width="40px" src="https://github.com/USSoccerFederation/ussf_ssac_23_soccer_gnn/blob/main/img/twitter.png?raw=true"/>](https://twitter.com/unravelsports)

🌀 Licenses
----
- For **commercial** use this repository falls under the [AGPL-3.0 License](LICENSE-COMMERCIAL)
- For **non-commerical** use this repository falls under the [BSD 3-Clause](LICENSE-NON-COMMERICIAL)
- Any **contributions** made to this repository fall under the [Apache License Version 2.0](LICENSE-3-CONTRIBUTING)

Because sports organizations (teams, clubs, federations) are a bit weird, I want to explicitely state that any (reseach) projects - that are not sold to third parties - within these _sports organizations_ explicitely fall under the **non-commerical** license.