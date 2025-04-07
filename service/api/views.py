from typing import List

import numpy as np
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
from rectools import Columns

from service.api.exceptions import UserNotFoundError
from service.log import app_logger
from service.models import Error

# Set auto_error=False so that missing auth is handled in our dependency
security = HTTPBearer(auto_error=False)


def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    request: Request = None,
) -> None:
    if not credentials:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid Authorization header")
    if request is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)
    if credentials.credentials != request.app.state.auth_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")


router = APIRouter(dependencies=[Depends(verify_token)])


class RecoResponse(BaseModel):
    user_id: int
    items: List[int]


@router.get(
    path="/health",
    tags=["Health"],
)
async def health() -> str:
    return "I am alive"


# Added helper functions to reduce cyclomatic complexity


def _get_reco_model_range(k_recs: int) -> List[int]:
    return list(range(k_recs))


def _get_reco_lightfm_4f(request: Request, user_id: int, k_recs: int) -> List[int]:
    required_attrs = ["lightfm_model", "lightfm_index", "dataset_with_features", "interactions", "user_embeddings"]
    if not all(hasattr(request.app.state, attr) for attr in required_attrs):
        raise HTTPException(status_code=500, detail="LightFM model not loaded properly")
    try:
        ds = request.app.state.dataset_with_features
        interactions = request.app.state.interactions
        if user_id not in ds.user_id_map.external_ids:
            app_logger.warning(f"User {user_id} not found in LightFM model")
            return [10440, 15297, 9728, 13865, 4151, 3734, 2657, 4880, 142, 6809]
        user_interactions = interactions[interactions[Columns.User] == user_id][Columns.Item].values
        user_idx = ds.user_id_map.convert_to_internal([user_id])[0]
        query_emb = request.app.state.user_embeddings[user_idx].reshape(1, -1)
        nbrs = request.app.state.lightfm_index.knnQueryBatch(query_emb, k=k_recs + len(user_interactions))
        reco_temp = ds.item_id_map.convert_to_external(nbrs[0][0])
        reco = np.array([reco_temp]) if isinstance(reco_temp, int) else np.array(reco_temp)
        return reco[np.isin(reco, user_interactions, invert=True)][:k_recs].tolist()
    except Exception as e:
        app_logger.error(f"Error getting recommendations for user {user_id} with LightFM: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting recommendations: {str(e)}")


def _get_reco_default(request: Request, user_id: int, model_name: str) -> List[int]:
    df = request.app.state.reco_models[model_name]
    user_data = df[df["user_id"] == user_id]
    if user_data.empty:
        return df[df["user_id"] == -9999]["item_id"].values.tolist()
    return user_data["item_id"].values.tolist()


@router.get(
    path="/reco/{model_name}/{user_id}",
    tags=["Recommendations"],
    response_model=RecoResponse,
    responses={404: {"model": List[Error], "description": "User or model not found"}},
)
async def get_reco(
    request: Request,
    model_name: str,
    user_id: int,
) -> RecoResponse:
    app_logger.info(f"Request for model: {model_name}, user_id: {user_id}")

    # Тут надо проверять, что юзер в числе тех, что есть в датасете,
    # если мы не готовы работать с холодными
    if user_id > 10**9:
        raise UserNotFoundError(error_message=f"User {user_id} not found")

    k_recs = request.app.state.k_recs
    app_logger.info(f"k_recs: {k_recs}")

    if model_name == "model_range":
        reco = _get_reco_model_range(k_recs)
    elif model_name == "lightfm_4f":
        reco = _get_reco_lightfm_4f(request, user_id, k_recs)
    elif model_name in request.app.state.reco_models:
        reco = _get_reco_default(request, user_id, model_name)
    else:
        raise HTTPException(status_code=404, detail="Model not found")

    return RecoResponse(user_id=user_id, items=reco)


def add_views(app: FastAPI) -> None:
    app.include_router(router)
