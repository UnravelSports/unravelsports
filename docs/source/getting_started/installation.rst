Installation
============

Basic Installation
------------------

The easiest way to install unravelsports is via pip:

.. code-block:: bash

   pip install unravelsports

This will install the core dependencies:

* ``kloppy>=3.18.0`` - For loading soccer tracking data
* ``polars[numpy]>=1.35.0`` - For fast data processing
* ``scipy>=1.0.0`` - For scientific computing

Requirements
------------

* Python 3.11, 3.12, or 3.13
* Operating System: Linux, macOS, or Windows

Optional Dependencies
---------------------

For Graph Neural Networks with PyTorch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want to use PyTorch Geometric for graph neural networks (recommended):

.. code-block:: bash

   pip install torch>=2.5.0
   pip install torch-geometric>=2.6.0
   pip install torchmetrics>=1.0.0
   pip install pytorch-lightning>=2.0.0

For Graph Neural Networks with Spektral (Python 3.11 only)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::
   Spektral support is only available on Python 3.11 and is considered deprecated.
   We recommend using PyTorch Geometric for new projects.

.. code-block:: bash

   pip install spektral==1.2.0
   pip install keras==2.14.0

   # For Intel/AMD processors:
   pip install tensorflow>=2.14.0

   # For Apple Silicon (M1/M2/M3):
   pip install tensorflow-macos>=2.14.0

Development Installation
------------------------

.. code-block:: bash

   git clone https://github.com/unravelsports/unravelsports.git
   cd unravelsports
   pip install -e .[test-torch]  # For Python 3.12+
   # OR
   pip install -e .[test]  # For Python 3.11 (includes Spektral)

Getting Help
------------

If you encounter any issues:

* Check the `GitHub Issues <https://github.com/unravelsports/unravelsports/issues>`_
* Open a new issue with details about your environment and error messages
