class MissingHomePlayerDataException(Exception):
    pass


class MissingAwayPlayerDataException(Exception):
    pass


class MissingBallDataException(Exception):
    pass


class InvalidAttackingTeamException(Exception):
    pass


class ModelParametersNotSetException(Exception):
    pass


class InvalidAttackingTeamTypeException(Exception):
    pass


class AdjcacenyMatrixTypeNotSetException(Exception):
    pass


class KeyMismatchException(Exception):
    pass


class SpektralDependencyError(ImportError):
    """Raised when Spektral or its dependencies are not properly installed."""

    def __init__(self):
        self.message = (
            "Seems like you don't have spektral installed.\n\n"
            "Requirements:\n"
            "  - Python 3.11 (recommended)\n\n"
            "Installation:\n"
            "  pip install spektral==1.2.0 keras==2.14.0 && "
            "(pip install tensorflow>=2.14.0 || pip install tensorflow-macos>=2.14.0)"
            "Warning:\n"
            "  If you want to use Spektral, it is advised to use unravelsports v1.1.0 or below."
            "  In general, it is advised to continue using PyTorch functionality instead"
        )
        super().__init__(self.message)
