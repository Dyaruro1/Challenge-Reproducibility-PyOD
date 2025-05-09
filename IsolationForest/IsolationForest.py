import numpy as np
import random
import multiprocessing as mp
from IsolationTreeNode import IsolationTreeNode

class IsolationForest:
    # Clase para detectar anomalias usando un bosque de arboles
    def __init__(self, n_estimators=100, subsample_size=256, max_height=None,
                 contamination=0.1, random_state=None, n_jobs=1):
        # Guarda los parametros y crea variables vacias
        self.n_estimators = n_estimators  # Numero de arboles
        self.subsample_size = subsample_size  # Tamano de muestra por arbol
        self.max_height = max_height  # Altura maxima de arboles
        self.contamination = contamination  # % de anomalias esperado
        self.random_state = random_state  # Semilla para reproducibilidad
        self.n_jobs = n_jobs  # Procesos paralelos
        self.trees = []  # Lista para guardar arboles
        self.threshold_ = None  # Umbral para decidir anomalias
        self.scores_ = None  # Puntuaciones de anomalia
        self.labels_ = None  # Etiquetas (1=anomalo, 0=normal)

    def fit(self, X):
        # Entrena el modelo con los datos X
        X = np.asarray(X)
        n_samples, n_features = X.shape

        # Fija la semilla si se dio
        if self.random_state is not None:
            random.seed(self.random_state)
            np.random.seed(self.random_state)

        # Ajusta tamano de muestra
        psi = self.subsample_size
        if psi is None or psi <= 0 or psi > n_samples:
            psi = n_samples
        self.psi = psi

        # Calcula altura maxima si no se dio
        if self.max_height is None:
            self.max_height = int(np.ceil(np.log2(self.psi)))

        # Prepara datos para cada arbol
        args = []
        seeds = np.random.randint(0, 1e9, size=self.n_estimators)
        for i in range(self.n_estimators):
            if psi < n_samples:
                indices = np.random.choice(n_samples, psi, replace=False)
                X_sample = X[indices]
            else:
                X_sample = X
            args.append((X_sample, 0, self.max_height, seeds[i]))

        # Construye arboles (en paralelo si n_jobs > 1)
        if self.n_jobs == 1:
            self.trees = [self._build_iTree_wrapper(arg) for arg in args]
        else:
            with mp.Pool(processes=self.n_jobs) as pool:
                self.trees = pool.map(self._build_iTree_wrapper, args)

        # Calcula puntuaciones, umbral y etiquetas
        self.scores_ = self.decision_function(X)
        self.threshold_ = np.percentile(self.scores_, 100 * (1 - self.contamination))
        self.labels_ = np.where(self.scores_ > self.threshold_, 1, 0)
        return self

    def _build_iTree_wrapper(self, args):
        # Prepara datos y semilla para construir un arbol
        X_sample, current_height, max_height, seed = args
        random.seed(int(seed))
        np.random.seed(int(seed))
        return self._build_iTree(X_sample, current_height, max_height)

    def _build_iTree(self, X, current_height, max_height):
        # Construye un arbol dividiendo datos al azar
        n_samples, n_features = X.shape
        # Si hay 1 muestra o se llego a max_height, crea nodo hoja
        if n_samples <= 1 or current_height >= max_height:
            return IsolationTreeNode(size=n_samples)
        # Si no hay variacion en datos, crea nodo hoja
        if np.all(np.std(X, axis=0) == 0):
            return IsolationTreeNode(size=n_samples)

        # Escoge caracteristica y valor de corte al azar
        q = random.randrange(n_features)
        min_q, max_q = X[:, q].min(), X[:, q].max()
        if min_q == max_q:
            return IsolationTreeNode(size=n_samples)

        p = random.uniform(min_q, max_q)
        left_mask = X[:, q] < p
        X_left = X[left_mask]
        X_right = X[~left_mask]
        # Si un grupo queda vacio, crea nodo hoja
        if X_left.shape[0] == 0 or X_right.shape[0] == 0:
            return IsolationTreeNode(size=n_samples)

        # Construye subarboles y crea nodo
        left_subtree = self._build_iTree(X_left, current_height + 1, max_height)
        right_subtree = self._build_iTree(X_right, current_height + 1, max_height)
        return IsolationTreeNode(feature_index=q, split_value=p,
                                 left=left_subtree, right=right_subtree)

    def _c_factor(self, psi):
        # Calcula factor para normalizar puntuaciones
        if psi <= 1:
            return 0.0
        elif psi == 2:
            return 1.0
        euler_gamma = 0.5772156649
        H = np.log(psi - 1) + euler_gamma
        return 2.0 * H - 2.0 * (psi - 1) / psi

    def decision_function(self, X):
        # Calcula puntuaciones de anomalia para cada dato
        X = np.asarray(X)
        n_samples, _ = X.shape
        scores = np.zeros(n_samples, dtype=float)

        for i in range(n_samples):
            x = X[i]
            # Calcula distancia promedio en todos los arboles
            path_lengths = np.array([tree.path_length(x, 0) for tree in self.trees])
            avg_path_length = path_lengths.mean()
            c = self._c_factor(self.psi)
            # Convierte distancia a puntuacion
            scores[i] = 2 ** (-avg_path_length / c) if c > 0 else 0.0

        return scores

    def predict(self, X):
        # Predice si los datos son anomalos (1) o normales (0)
        scores = self.decision_function(X)
        return np.where(scores > self.threshold_, 1, 0)

    def fit_predict(self, X):
        # Entrena el modelo y predice etiquetas
        self.fit(X)
        return self.labels_

    def score_samples(self, X):
        # Otro nombre para decision_function
        return self.decision_function(X)