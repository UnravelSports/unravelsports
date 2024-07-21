from spektral.data import Graph
from scipy.sparse import csr_matrix, isspmatrix
import numpy as np


class SpektralGraph(Graph):
    def __init__(self, x=None, a=None, e=None, y=None, **kwargs):
        super().__init__(x=x, a=a, e=e, y=y, **kwargs)
        self._extra_attributes = kwargs

    def get_config(self):
        config = {
            "x": self.x.tolist() if self.x is not None else None,
            "a": self._serialize_sparse(self.a),
            "e": self.e.tolist() if self.e is not None else None,
            "y": self.y.tolist() if self.y is not None else None,
        }
        config.update(self._extra_attributes)
        return config

    @classmethod
    def from_config(cls, config):
        x = np.array(config.pop("x")) if config.get("x") is not None else None
        a = cls._deserialize_sparse(config.pop("a"))
        e = np.array(config.pop("e")) if config.get("e") is not None else None
        y = np.array(config.pop("y")) if config.get("y") is not None else None
        return cls(x=x, a=a, e=e, y=y, **config)

    def _get_extra_attributes(self):
        extra_attributes = {
            k: v
            for k, v in self.__dict__.items()
            if k not in ["x", "a", "e", "y", "_extra_attributes"]
        }
        return extra_attributes

    @staticmethod
    def _serialize_sparse(matrix):
        if isspmatrix(matrix):
            return {
                "data": matrix.data.tolist(),
                "indices": matrix.indices.tolist(),
                "indptr": matrix.indptr.tolist(),
                "shape": matrix.shape,
            }
        return matrix

    @staticmethod
    def _deserialize_sparse(serialized):
        if serialized and all(
            k in serialized for k in ("data", "indices", "indptr", "shape")
        ):
            return csr_matrix(
                (serialized["data"], serialized["indices"], serialized["indptr"]),
                shape=serialized["shape"],
            )
        return serialized
