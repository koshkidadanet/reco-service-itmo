import asyncio
import pickle
from concurrent.futures.thread import ThreadPoolExecutor
from typing import Any, Dict

import nmslib
import numpy as np
import pandas as pd
import uvloop
from fastapi import FastAPI
from rectools.models import LightFMWrapperModel

from ..log import app_logger, setup_logging
from ..settings import ServiceConfig
from .exception_handlers import add_exception_handlers
from .middlewares import add_middlewares
from .views import add_views

__all__ = ("create_app",)


def setup_asyncio(thread_name_prefix: str) -> None:
    uvloop.install()

    loop = asyncio.get_event_loop()

    executor = ThreadPoolExecutor(thread_name_prefix=thread_name_prefix)
    loop.set_default_executor(executor)

    def handler(_, context: Dict[str, Any]) -> None:
        message = "Caught asyncio exception: {message}".format_map(context)
        app_logger.warning(message)

    loop.set_exception_handler(handler)


def create_app(config: ServiceConfig) -> FastAPI:
    setup_logging(config)
    setup_asyncio(thread_name_prefix=config.service_name)

    app = FastAPI(debug=False)
    app.state.k_recs = config.k_recs
    app.state.auth_token = config.auth_token

    app.state.reco_models = {}
    try:
        model_path = "artifacts/reco.parquet"
        app.state.reco_models["userknn_pop"] = pd.read_parquet(model_path, engine="pyarrow")
        app_logger.info("Successfully loaded userknn_pop")
    except (FileNotFoundError, pd.errors.ParserError, OSError) as e:
        app_logger.error(f"Error loading userknn_pop: {e}")

    try:
        app.state.lightfm_model = LightFMWrapperModel.load("artifacts/ligftfm_4f")

        app.state.lightfm_index = nmslib.init(  # pylint: disable=I1101
            method="hnsw", space="cosinesimil"
        )
        app.state.lightfm_index.loadIndex("artifacts/ligftfm_4f__hnsw_index")

        with open("artifacts/dataset_with_features", "rb") as f:
            app.state.dataset_with_features = pickle.load(f)

        app.state.interactions = app.state.dataset_with_features.get_raw_interactions()

        user_embeddings, _ = app.state.lightfm_model.get_vectors(app.state.dataset_with_features)
        extra_zero = np.zeros((user_embeddings.shape[0], 1))
        app.state.user_embeddings = np.append(user_embeddings, extra_zero, axis=1)

        app_logger.info("Successfully loaded lightfm model and components")
    except Exception as e:  # pylint: disable=broad-exception-caught
        app_logger.error(f"Error loading lightfm model: {e}")

    add_views(app)
    add_middlewares(app)
    add_exception_handlers(app)

    return app
