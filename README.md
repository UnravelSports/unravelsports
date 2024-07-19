![unravelsports logo](https://github.com/UnravelSports/unravelsports.github.io/blob/main/imgs/unravelsports-5500x800-3.png?raw=true)
<div align="right">

[![Powered by PySport](https://img.shields.io/badge/powered%20by-PySport-orange.svg?style=flat&colorA=104467&colorB=007D8A)](https://pysport.org)
</div>

🌀 `pip install unravelsports`


🌀 What is it?
-----

The **unravelsports** package aims to aid researchers, analysts and enthusiasts by providing intermediary steps in the complex process of turning raw sports data into meaningful information and actionable insights.

🌀 Features
-----

- ⚽ Converting **positional soccer data** into graphs to train **graph neural networks** by leveraging the powerful [**Kloppy**](https://github.com/PySport/kloppy/tree/master) data conversion standard and [**Spektral**](https://github.com/danielegrattarola/spektral) - a flexible framework for creating graph neural networks. 
- ⚽ Randomizing and splitting data into **train, test and validation sets** along matches, sequences or possessions to avoid leakage and improve model quality.
- ⚽ Due to the power of **Kloppy**, **unravelsports** supports these actions for _Metrica_, _Sportec_, _Tracab (CyronHego)_, _SecondSpectrum_, _SkillCorner_ and _StatsPerform_ tracking data.

🌀 Getting Started
-----
This [**Getting Started Jupyter Notebook**](examples/getting_started.ipynb) explains how to convert any positional tracking data from **Kloppy** to **Spektral GNN** in a few easy steps while walking you through the most important features and documentation.

🌀 Documentation
-----
For now, follow the Getting Started, more documentation will follow!

Additional reading:
- 📖 [A Graph Neural Network Deep-dive into Successful Counterattacks {A. Sahasrabudhe & J. Bekkers, 2023}](https://github.com/USSoccerFederation/ussf_ssac_23_soccer_gnn/tree/main)

🌀 Installation
----
**unravelsports** is compatible with Python 3.10+ and it is tested on the latest versions of Ubuntu, MacOS and Windows.

The easiest way to get started is by running:

```bash
pip install unravelsports
```

🌀 Contributing
----
All contributions, bug reports, bug fixes, documentation improvements, enhancements, and ideas are welcome.

An overview on how to contribute can be found in the [**contributing guide**](CONTRIBUTING.md).

🌀 Citation
----
If you use this repository for any research or public project, please reference:

[Sahasrabudhe, A., & Bekkers, J. (2023). A graph neural network deep-dive into successful counterattacks. MIT Sloan Sports Analytics Conference.](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=eerdAe8AAAAJ&citation_for_view=eerdAe8AAAAJ:d1gkVwhDpl0C)

```bash
@inproceedings{sahasrabudhe2023graph,
  title={A Graph Neural Network deep-dive into successful counterattacks},
  author={Sahasrabudhe, Amod and Bekkers, Joris},
  booktitle={17th Annual MIT Sloan Sports Analytics Conference. Boston, MA, USA: MIT},
  pages={15},
  year={2023}
}
```