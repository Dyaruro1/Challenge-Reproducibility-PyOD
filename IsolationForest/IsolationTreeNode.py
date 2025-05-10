import numpy as np
class IsolationTreeNode:
    """
    Nodo para el arbol de aislamiento.
    Nodo interno con una caracteristica (feature) y valor de corte (split), o hoja si no tiene hijos.
    """
    def __init__(self, feature_index=None, split_value=None, left=None, right=None, size=0):
        self.feature = feature_index  # indice de la caracteristica usada en la division
        self.split = split_value     # valor de corte aleatorio en esa caracteristica
        self.left = left             # subarbol izquierdo (valores < split)
        self.right = right           # subarbol derecho (valores >= split)
        self.size = size             # tamano de la muestra en la hoja (solo para nodo hoja)

    def is_external(self):
        return self.left is None and self.right is None

    def path_length(self, x, current_depth=0):
        """
        Calcula la longitud del camino desde este nodo hasta la hoja donde x queda aislado.
        """
        if self.is_external():
            return current_depth + self._external_path_length()
        if x[self.feature] < self.split:
            return self.left.path_length(x, current_depth + 1)
        else:
            return self.right.path_length(x, current_depth + 1)

    def _external_path_length(self):
        """
        Calculo del valor esperado de la longitud del camino en una hoja, basado en su tamano.
        """
        if self.size <= 1:
            return 0
        else:
            return 2 * np.log(self.size - 1) + 0.5772156649 - (2 * (self.size - 1) / self.size)