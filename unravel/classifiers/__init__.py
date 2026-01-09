try:
    from .crystal_graph import CrystalGraphClassifier

    __all__ = ["CrystalGraphClassifier"]
except ImportError:

    class CrystalGraphClassifier:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "CrystalGraphClassifier requires spektral (Python 3.11 only). "
                "Install with: pip install spektral==1.2.0 keras==2.14.0 tensorflow>=2.14.0"
            )

    __all__ = ["CrystalGraphClassifier"]


try:
    from .crystal_graph_pyg import PyGLightningCrystalGraphClassifier

    __all__.append("PyGLightningCrystalGraphClassifier")
except ImportError:

    class PyGLightningCrystalGraphClassifier:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyGLightningCrystalGraphClassifier requires PyTorch and PyTorch Geometric. "
                "Install with: pip install torch torch-geometric pytorch-lightning torchmetrics"
            )

    __all__.append("PyGLightningCrystalGraphClassifier")
