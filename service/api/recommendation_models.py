# pylint: disable=too-many-instance-attributes,too-many-return-statements,
# pylint: disable=broad-exception-caught

import json
import logging
import pickle
from typing import Any, Dict, List, Optional

import nmslib
import numpy as np
import pandas as pd
import torch
from rectools.columns import Columns
from rectools.models import LightFMWrapperModel

POPULAR_ITEMS = [10440, 15297, 9728, 13865, 4151, 3734, 2657, 4880, 142, 6809]


class PopularModel:
    """A model that returns a fixed list of popular items as recommendations"""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger
        self.popular_items = POPULAR_ITEMS
        if self.logger:
            self.logger.info("Successfully initialized PopularModel")

    def recommend(self, user_id: int, k_recs: int) -> List[int]:
        """Return popular item recommendations."""
        if self.logger:
            self.logger.debug(f"Popular recommendations for user {user_id}")
        return self.popular_items[:k_recs]


class UserKnnPopModel:
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.recomendations_df: Optional[pd.DataFrame] = None
        self.logger = logger
        self._load_model()

    def _load_model(self) -> None:
        try:
            recomendations_path = "artifacts/reco.parquet"
            self.recomendations_df = pd.read_parquet(recomendations_path, engine="pyarrow")
            if self.logger:
                self.logger.info("Successfully loaded userknn_pop")
        except (FileNotFoundError, pd.errors.ParserError, OSError) as e:
            if self.logger:
                self.logger.error(f"Error loading userknn_pop: {e}")

    def recommend(self, user_id: int, k_recs: int) -> List[int]:
        user_data = self.recomendations_df[self.recomendations_df["user_id"] == user_id]
        if user_data.empty:
            return self.recomendations_df[self.recomendations_df["user_id"] == -9999]["item_id"].values.tolist()
        return user_data["item_id"].values.tolist()[:k_recs]


class LightFMModel:
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.model: Optional[LightFMWrapperModel] = None
        self.index: Optional[Any] = None
        self.dataset_with_features: Optional[Any] = None
        self.interactions: Optional[pd.DataFrame] = None
        self.user_embeddings: Optional[np.ndarray] = None
        self.logger = logger
        self._load_model()

    def _load_model(self) -> None:
        try:
            self.model = LightFMWrapperModel.load("artifacts/ligftfm_4f")

            self.index = nmslib.init(method="hnsw", space="cosinesimil")
            if self.index is not None:
                self.index.loadIndex("artifacts/ligftfm_4f__hnsw_index")

            with open("artifacts/dataset_with_features", "rb") as f:
                self.dataset_with_features = pickle.load(f)

            if self.dataset_with_features is not None:
                self.interactions = self.dataset_with_features.get_raw_interactions()

                if self.model is not None:
                    user_embeddings, _ = self.model.get_vectors(self.dataset_with_features)
                    extra_zero = np.zeros((user_embeddings.shape[0], 1))
                    self.user_embeddings = np.append(user_embeddings, extra_zero, axis=1)

            if self.logger:
                self.logger.info("Successfully loaded lightfm model and components")
        except Exception as e:  # pylint: disable=broad-exception-caught
            if self.logger:
                self.logger.error(f"Error loading lightfm model: {e}")

    def _is_model_ready(self) -> bool:
        """Check if all necessary model components are available."""
        return (
            self.dataset_with_features is not None
            and self.interactions is not None
            and self.user_embeddings is not None
            and self.index is not None
            and self.model is not None
        )

    def recommend(self, user_id: int, k_recs: int) -> List[int]:
        """Recommendations using LightFM model or pop items."""
        if not self._is_model_ready():
            if self.logger:
                self.logger.warning("Model components are missing, returning popular items")
            return POPULAR_ITEMS[:k_recs]

        if user_id not in self.dataset_with_features.user_id_map.external_ids:
            if self.logger:
                self.logger.warning(f"User {user_id} not found in LightFM model")
            return POPULAR_ITEMS[:k_recs]

        user_interactions = self.interactions[self.interactions[Columns.User] == user_id][Columns.Item].values
        user_idx = self.dataset_with_features.user_id_map.convert_to_internal([user_id])[0]

        query_emb = self.user_embeddings[user_idx].reshape(1, -1)
        nbrs = self.index.knnQueryBatch(query_emb, k=k_recs + len(user_interactions))

        reco_temp = self.dataset_with_features.item_id_map.convert_to_external(nbrs[0][0])
        reco = np.array([reco_temp]) if isinstance(reco_temp, int) else np.array(reco_temp)

        return reco[np.isin(reco, user_interactions, invert=True)][:k_recs].tolist()


class RangeModel:
    """
    A simple model that returns sequential integers as recommendations.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger
        if self.logger:
            self.logger.info("Successfully initialized RangeModel")

    def recommend(self, user_id: int, k_recs: int) -> List[int]:
        if self.logger:
            self.logger.debug(f"Generating range recommendations for user {user_id} with k={k_recs}")
        return list(range(k_recs))


class ItemModel(torch.nn.Module):
    def __init__(self, n_factors: int = 128, dropout: float = 0.2):
        super().__init__()
        self.fc1 = torch.nn.Linear(842, n_factors)
        self.ln1 = torch.nn.LayerNorm(n_factors)
        self.fc2 = torch.nn.Linear(n_factors, n_factors)
        self.dropout = torch.nn.Dropout(dropout)
        self.fc3 = torch.nn.Linear(n_factors, n_factors)

    def forward(self, x):
        x = torch.relu(self.ln1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = x + torch.relu(self.fc2(x))
        x = self.fc3(x)
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        return x


class UserModel(torch.nn.Module):
    def __init__(self, n_factors: int = 128, dropout: float = 0.2):
        super().__init__()
        self.fc1_meta = torch.nn.Linear(16, n_factors)
        self.ln_meta = torch.nn.LayerNorm(n_factors)
        self.fc2_meta = torch.nn.Linear(n_factors, n_factors)
        self.dropout = torch.nn.Dropout(dropout)
        self.fc1_interaction = torch.nn.Linear(12320, n_factors)
        self.ln_inter = torch.nn.LayerNorm(n_factors)
        self.fc2_inter = torch.nn.Linear(n_factors, n_factors)
        self.fc3 = torch.nn.Linear(n_factors * 2, n_factors)

    def forward(self, meta, interaction):
        meta = torch.relu(self.ln_meta(self.fc1_meta(meta)))
        meta = self.dropout(meta)
        meta = torch.relu(self.fc2_meta(meta))
        meta = meta + torch.relu(self.fc2_meta(meta))
        interaction = torch.relu(self.ln_inter(self.fc1_interaction(interaction)))
        interaction = self.dropout(interaction)
        interaction = torch.relu(self.fc2_inter(interaction))
        x = torch.cat([meta, interaction], dim=1)
        x = self.fc3(x)
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        return x


class DSSMModel:
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger: Optional[logging.Logger] = logger
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.k_recs_default: int = 10

        self.N_FACTORS: int = 128
        self.ITEM_MODEL_SHAPE: List[int] = [842]
        self.USER_META_MODEL_SHAPE: List[int] = [16]
        self.USER_INTERACTION_MODEL_SHAPE: List[int] = [12320]

        self.i2v: Optional[ItemModel] = None
        self.u2v: Optional[UserModel] = None
        self.item_id_to_iid: Optional[Dict[int, int]] = None
        self.user_id_to_uid: Optional[Dict[int, int]] = None
        self.iid_to_item_id: Optional[Dict[int, int]] = None
        self.uid_to_user_id: Optional[Dict[int, int]] = None
        self.interactions: Optional[pd.DataFrame] = None
        self.index: Optional[Any] = None
        self.users_vec: Optional[np.ndarray] = None

        self._load_model()

    def _load_model(self) -> None:
        try:
            self.i2v = ItemModel(self.N_FACTORS).to(self.device)
            self.i2v.load_state_dict(torch.load("artifacts/i2v_model_bpr_loss_norm_full", map_location=self.device))
            self.i2v.eval()

            self.u2v = UserModel(self.N_FACTORS).to(self.device)
            self.u2v.load_state_dict(torch.load("artifacts/u2v_model_bpr_loss_norm_full", map_location=self.device))
            self.u2v.eval()

            with open("artifacts/item_id_to_iid.json", "r", encoding="utf-8") as f:
                item_id_to_iid_raw = json.load(f)
            with open("artifacts/user_id_to_uid.json", "r", encoding="utf-8") as f:
                user_id_to_uid_raw = json.load(f)
            with open("artifacts/iid_to_item_id.json", "r", encoding="utf-8") as f:
                iid_to_item_id_raw = json.load(f)
            with open("artifacts/uid_to_user_id.json", "r", encoding="utf-8") as f:
                uid_to_user_id_raw = json.load(f)

            self.item_id_to_iid = {int(k): int(v) for k, v in item_id_to_iid_raw.items()}
            self.user_id_to_uid = {int(k): int(v) for k, v in user_id_to_uid_raw.items()}
            self.iid_to_item_id = {int(k): int(v) for k, v in iid_to_item_id_raw.items()}
            self.uid_to_user_id = {int(k): int(v) for k, v in uid_to_user_id_raw.items()}

            self.interactions = pd.read_parquet("artifacts/interactions.parquet")

            self.index = nmslib.init(method="hnsw", space="cosinesimil")
            self.index.loadIndex("artifacts/bpr_loss_norm__hnsw_index")

            self.users_vec = np.load("artifacts/users_vec.npy")

            if self.logger:
                self.logger.info("Successfully loaded DSSM model and data")
        except Exception as e:
            self.i2v = None
            self.u2v = None
            self.item_id_to_iid = None
            self.user_id_to_uid = None
            self.iid_to_item_id = None
            self.uid_to_user_id = None
            self.interactions = None
            self.index = None
            self.users_vec = None
            if self.logger:
                self.logger.error(f"Error loading DSSM model: {e}")

    def _is_model_ready(self) -> bool:
        return (
            self.i2v is not None
            and self.u2v is not None
            and self.item_id_to_iid is not None
            and self.user_id_to_uid is not None
            and self.iid_to_item_id is not None
            and self.uid_to_user_id is not None
            and self.interactions is not None
            and self.index is not None
            and self.users_vec is not None
        )

    def recommend(self, user_id: int, k_recs: int) -> List[int]:
        if not self._is_model_ready():
            if self.logger:
                self.logger.warning("DSSM model components are missing, returning popular items")
            return POPULAR_ITEMS[: self.k_recs_default]
        # Если пользователь неизвестен — возвращаем популярные
        if self.user_id_to_uid is None or user_id not in self.user_id_to_uid:
            if self.logger:
                self.logger.warning(f"User {user_id} not found in DSSM model, returning popular items")
            return POPULAR_ITEMS[: self.k_recs_default]

        uid = self.user_id_to_uid[user_id]
        if self.interactions is None:
            return POPULAR_ITEMS[: self.k_recs_default]
        user_interactions = self.interactions[self.interactions["user_id"] == user_id]["item_id"].values
        if self.users_vec is None:
            return POPULAR_ITEMS[: self.k_recs_default]
        user_vec = self.users_vec[uid]
        user_vec = np.atleast_2d(user_vec)  # Ensure 2D shape for nmslib
        if self.index is None:
            return POPULAR_ITEMS[: self.k_recs_default]
        nbrs = self.index.knnQueryBatch(user_vec, k=k_recs + len(user_interactions))

        if self.iid_to_item_id is None:
            reco_temp = nbrs[0][0]
        else:
            reco_temp = [self.iid_to_item_id.get(int(x), int(x)) for x in nbrs[0][0]]
        reco = np.array(reco_temp)
        reco = reco[np.isin(reco, user_interactions, invert=True)][:k_recs]
        return reco.tolist()
