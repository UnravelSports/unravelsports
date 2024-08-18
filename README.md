![unravelsports logo](https://github.com/UnravelSports/unravelsports.github.io/blob/main/imgs/unravelsports-5500x800-4.png?raw=true)
<div align="right">

[![Powered by PySport](https://img.shields.io/badge/powered%20by-PySport-orange.svg?style=flat&colorA=C3C3C3&colorB=4B99CC)](https://pysport.org) 
[![Powered by PySport](https://img.shields.io/badge/powered%20by-UnravelSports-orange.svg?style=flat&colorB=E6B611&colorA=C3C3C3)](https://unravelsports.github.io/)
[![tl;dr legal](https://img.shields.io/badge/license-Mozilla%20Public%20License%20v2.0-orange.svg?style=flat&colorA=C3C3C3&colorB=E20E6A)](https://www.tldrlegal.com/license/mozilla-public-license-2-0-mpl-2) 
</div>

🌀 `pip install unravelsports`


🌀 What is it?
-----

The **unravelsports** package aims to aid researchers, analysts and enthusiasts by providing intermediary steps in the complex process of turning raw sports data into meaningful information and actionable insights.

🌀 Features
-----

⚽ Convert **positional soccer data** into graphs to train **graph neural networks** by leveraging the powerful [**Kloppy**](https://github.com/PySport/kloppy/tree/master) data conversion standard and [**Spektral**](https://github.com/danielegrattarola/spektral) - a flexible framework for training graph neural networks. 

⚽ **Randomize** and **split** data into train, test and validation sets along matches, sequences or possessions to avoid leakage and improve model quality.

⚽ **Train**, **validate** and **test** your (custom) Graph model(s) and easily **predict** on new data.

⚽ Leverage the power of **Kloppy** standardization and **unravelsports** to execute the these features for _Metrica_, _Sportec_, _Tracab (CyronHego)_, _SecondSpectrum_, _SkillCorner_ and _StatsPerform_ tracking data.

⌛ ***More to come soon...!***

🌀 Quick Start
-----
📖 The [**Quick Start Jupyter Notebook**](examples/0_quick_start_guide.ipynb) explains how to convert any positional tracking data from **Kloppy** to **Spektral GNN** in a few easy steps while walking you through the most important features and documentation.

📖 The [**Graph Converter Tutorial Jupyter Notebook**](examples/1_kloppy_gnn_train.ipynb) gives an in-depth walkthrough.

🌀 Documentation
-----
For now, follow the [**Graph Converter Tutorial**](examples/1_kloppy_gnn_train.ipynb), more documentation will follow!

Additional reading:

📖 [A Graph Neural Network Deep-dive into Successful Counterattacks {A. Sahasrabudhe & J. Bekkers, 2023}](https://github.com/USSoccerFederation/ussf_ssac_23_soccer_gnn/tree/main)

🌀 Installation
----
The easiest way to get started is:

```bash
pip install unravelsports
```

Due to compatibility issues **unravelsports** currently only works on Python 3.11 with:
```
spektral==1.20.0 
tensorflow==2.14.0 
keras==2.14.0
kloppy==3.15.0
```
These dependencies come pre-installed with the package. It is advised to create a [virtual environment](https://virtualenv.pypa.io/en/latest/).

This package is tested on the latest versions of Ubuntu, MacOS and Windows. 

🌀 Contributing
----
All contributions, bug reports, bug fixes, documentation improvements, enhancements, and ideas are welcome. 

An overview on how to contribute can be found in the [**contributing guide**](CONTRIBUTING.md).

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
  version = {0.1.0},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/unravelsports/unravelsports}
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
This project is licensed under the [Mozilla Public License Version 2.0 (MPL)](LICENSE), which requires that you include a copy of the license and provide attribution to the original authors. Any modifications you make to the MPL-licensed files must be documented, and the source code for those modifications must be made open-source under the same license.

