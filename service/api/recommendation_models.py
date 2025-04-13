import logging
import pickle
from typing import Any, List, Optional

import nmslib
import numpy as np
import pandas as pd
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

            self.index = nmslib.init(method="hnsw", space="cosinesimil")  # pylint: disable=c-extension-no-member
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
    