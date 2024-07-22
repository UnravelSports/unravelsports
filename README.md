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

<ul style="list-style: none; padding: 0; margin-left: 1.2em;">
  <li style="margin-bottom: 8px;">
    <span style="display: inline-block; width: 1.2em; margin-right: 0.5em;">⚽</span>
    Convert <strong>positional soccer data</strong> into graphs to train <strong>graph neural networks</strong> by leveraging the powerful <a href="https://github.com/PySport/kloppy/tree/master"><strong>Kloppy</strong></a> data conversion standard and <a href="https://github.com/danielegrattarola/spektral"><strong>Spektral</strong></a> - a flexible framework for creating GNNs.
  </li>
  <li style="margin-bottom: 8px;">
    <span style="display: inline-block; width: 1.2em; margin-right: 0.5em;">⚽</span>
    Randomize and split data into <strong>train, test and validation sets</strong> along matches, sequences or possessions to avoid leakage and improve model quality.
  </li>
  <li style="margin-bottom: 8px;">
    <span style="display: inline-block; width: 1.2em; margin-right: 0.5em;">⚽</span>
    Due to the power of <strong>Kloppy</strong>, <strong>unravelsports</strong> supports these actions for <em>Metrica</em>, <em>Sportec</em>, <em>Tracab (CyronHego)</em>, <em>SecondSpectrum</em>, <em>SkillCorner</em> and <em>StatsPerform</em> tracking data.
  </li>
  <li style="margin-bottom: 8px;">
    <span style="display: inline-block; width: 1.2em; margin-right: 0.5em;">⏳</span>
    <strong>More to come...</strong>
  </li>
</ul>

🌀 Getting Started
-----
<ul style="list-style: none; padding: 0; margin-left: 1.2em;">
  <li style="margin-bottom: 8px;">
    <span style="display: inline-block; width: 1.2em; margin-right: 0.5em;">📖</span>
    The <a href="examples/0_getting_started.ipynb"><strong>Getting Started Jupyter Notebook</strong></a> explains how to convert any positional tracking data from <strong>Kloppy</strong> to <strong>Spektral GNN</strong> in a few easy steps while walking you through the most important functionality.
  </li>
  <li style="margin-bottom: 8px;">
    <span style="display: inline-block; width: 1.2em; margin-right: 0.5em;">📖</span>
    The <a href="examples/1_tutorial_graph_converter.ipynb"><strong>Graph Converter Tutorial Jupyter Notebook</strong></a> gives an in-depth walkthrough.
  </li>
</ul>

🌀 Documentation
-----
For now, follow the [**Graph Converter Tutorial**](examples/1_tutorial_graph_converter.ipynb), more documentation will follow!

Additional reading:
<ul style="list-style: none; padding: 0; margin-left: 1.2em;">
  <li style="margin-bottom: 8px;">
    <span style="display: inline-block; width: 1.2em; margin-right: 0.5em;">📖</span>
    <a href="https://github.com/USSoccerFederation/ussf_ssac_23_soccer_gnn/tree/main"><strong>A Graph Neural Network Deep-dive into Successful Counterattacks {A. Sahasrabudhe & J. Bekkers, 2023}</strong></a>
  </li>
</ul>

🌀 Installation
----
**unravelsports** is compatible with Python 3.10+ and it is tested on the latest versions of Ubuntu, MacOS and Windows.

The easiest way to get started is by running:

```bash
pip install unravelsports
```

🌀 Contributing
----
All contributions, bug reports, bug fixes, documentation improvements, enhancements, and ideas are welcome. Feel free to create a Pull Request for any improvements you make that do not contribute to winning more games!

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

Because sports organizations (teams, clubs, federations) are a bit weird, I want to explicitely state that any (reseach) projects - that are not sold to third parties - within sports organizations explicitely fall under the **non-commerical** license.