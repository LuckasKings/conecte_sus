# -*- coding: utf-8 -*-
"""
Pré-processador Clínico Híbrido V5 para SIH/SUS:
Combina Top-N One-Hot Encoding para categorias CID-10 de 3 dígitos (`cid3`) com
Target Encoding Bayesiano Hierárquico (Empirical Bayes Shrinkage com fallback para `capitulo_cid`),
eliminando ruído na cauda rara e prevenindo overfitting via Out-of-Fold no treino.
"""

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_CAT_COLS = [
    "sexo",
    "capitulo_cid",
    "especialidade_leito",
    "carater_internacao",
    "complexidade",
    "transferido_intermun",
    "internacao_fim_de_semana",
]


def _extrair_cid3_limpo(df: pd.DataFrame) -> np.ndarray:
    """Extrai o código cid3 padronizado (3 caracteres) ou '_CAPITULO_' quando ausente."""
    if "cid3" in df.columns:
        raw = df["cid3"].fillna("_CAPITULO_").astype(str).str.strip()
    elif "diag_princ" in df.columns:
        raw = df["diag_princ"].fillna("").astype(str).str.strip().str[:3]
    else:
        return np.full(len(df), "_CAPITULO_", dtype=object)
    raw = raw.where(~raw.isin(["", "nan", "None", "IGN", "_CAPITULO_"]), "_CAPITULO_")
    return raw.str[:3].where(raw != "_CAPITULO_", "_CAPITULO_").values


class AdmissionV5Preprocessor(BaseEstimator, TransformerMixin):
    """
    Transforma variáveis da admissão hospitalar (9 entradas = 8 base + cid3) em matriz esparsa float32.
    - `idade_anos`: padronizada via StandardScaler
    - `te_cid3_bayes`: risco suavizado de `cid3` com prior hierárquico por `capitulo_cid` (OOF no treino)
    - OneHotEncoder: 7 categóricas base + `cid3_top` (Top-N categorias mais frequentes + OTHER)
    """

    def __init__(self, top_n_cid3: int = 200, bayes_m: float = 50.0, n_folds: int = 5, random_state: int = 42):
        self.top_n_cid3 = top_n_cid3
        self.bayes_m = bayes_m
        self.n_folds = n_folds
        self.random_state = random_state

    def _compute_bayes_maps(self, cap_arr: np.ndarray, cid_arr: np.ndarray, y_arr: np.ndarray):
        g_mean = float(np.mean(y_arr))
        df_tmp = pd.DataFrame({"cap": cap_arr, "cid": cid_arr, "y": y_arr})
        cap_means = df_tmp.groupby("cap")["y"].mean().to_dict()
        mask_valid = df_tmp["cid"] != "_CAPITULO_"
        agg = df_tmp.loc[mask_valid].groupby("cid").agg(
            n=("y", "count"),
            s=("y", "sum"),
            cap=("cap", "first"),
        )
        priors = agg["cap"].map(cap_means).fillna(g_mean)
        cid_map = ((agg["s"] + self.bayes_m * priors) / (agg["n"] + self.bayes_m)).to_dict()
        return g_mean, cap_means, cid_map

    def _encode_bayes_with_maps(self, cap_arr: np.ndarray, cid_arr: np.ndarray, g_mean: float, cap_means: dict, cid_map: dict) -> np.ndarray:
        priors = np.array([cap_means.get(c, g_mean) for c in cap_arr], dtype=np.float32)
        out = np.array(
            [cid_map.get(cid, pr) if cid != "_CAPITULO_" else pr for cid, pr in zip(cid_arr, priors)],
            dtype=np.float32,
        )
        return out.reshape(-1, 1)

    def _prep_cat_frame(self, df: pd.DataFrame, cid_arr: np.ndarray) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for col in BASE_CAT_COLS:
            out[col] = df[col].astype(str)
        out["cid3_top"] = np.where(np.isin(cid_arr, list(self.top_cids_)), cid_arr, "OTHER")
        return out

    def fit(self, X: pd.DataFrame, y):
        y_arr = np.asarray(y, dtype=np.float32)
        cid_arr = _extrair_cid3_limpo(X)
        cap_arr = X["capitulo_cid"].astype(str).values

        # Top-N CID3 (exclui placeholder _CAPITULO_)
        vc = pd.Series(cid_arr[cid_arr != "_CAPITULO_"]).value_counts()
        self.top_cids_ = set(vc.head(self.top_n_cid3).index.tolist())

        # Scaler numérico
        self.scaler_ = StandardScaler()
        self.scaler_.fit(X[["idade_anos"]].values.astype(np.float32))

        # One-Hot nas 7 categóricas + cid3_top
        cat_df = self._prep_cat_frame(X, cid_arr)
        self.ohe_ = OneHotEncoder(handle_unknown="ignore", dtype=np.float32)
        self.ohe_.fit(cat_df)

        # Mapas Bayesianos finais usando todo o conjunto de treino
        self.global_mean_, self.cap_means_, self.cid_map_ = self._compute_bayes_maps(cap_arr, cid_arr, y_arr)
        return self

    def fit_transform(self, X: pd.DataFrame, y=None):
        if y is None:
            return self.transform(X)

        y_arr = np.asarray(y, dtype=np.float32)
        self.fit(X, y_arr)

        cid_arr = _extrair_cid3_limpo(X)
        cap_arr = X["capitulo_cid"].astype(str).values

        # Out-of-Fold Bayesian Target Encoding para evitar data leakage dentro do treino
        oof_te = np.zeros((len(X), 1), dtype=np.float32)
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        for tr_i, va_i in skf.split(X, y_arr.astype(int)):
            g_f, cap_f, cid_f = self._compute_bayes_maps(cap_arr[tr_i], cid_arr[tr_i], y_arr[tr_i])
            oof_te[va_i] = self._encode_bayes_with_maps(cap_arr[va_i], cid_arr[va_i], g_f, cap_f, cid_f)

        age_scaled = self.scaler_.transform(X[["idade_anos"]].values.astype(np.float32)).astype(np.float32)
        cat_mat = self.ohe_.transform(self._prep_cat_frame(X, cid_arr))
        return sparse.hstack([sparse.csr_matrix(age_scaled), sparse.csr_matrix(oof_te), cat_mat], format="csr")

    def transform(self, X: pd.DataFrame):
        cid_arr = _extrair_cid3_limpo(X)
        cap_arr = X["capitulo_cid"].astype(str).values
        age_scaled = self.scaler_.transform(X[["idade_anos"]].values.astype(np.float32)).astype(np.float32)
        te_col = self._encode_bayes_with_maps(cap_arr, cid_arr, self.global_mean_, self.cap_means_, self.cid_map_)
        cat_mat = self.ohe_.transform(self._prep_cat_frame(X, cid_arr))
        return sparse.hstack([sparse.csr_matrix(age_scaled), sparse.csr_matrix(te_col), cat_mat], format="csr")

    def get_feature_names_out(self, input_features=None):
        ohe_names = list(self.ohe_.get_feature_names_out())
        return np.array(["idade_anos", "te_cid3_bayes"] + ohe_names)
