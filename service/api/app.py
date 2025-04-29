import asyncio
from concurrent.futures.thread import ThreadPoolExecutor
from typing import Any, Dict

import uvloop
from fastapi import FastAPI

from ..log import app_logger, setup_logging
from ..settings import ServiceConfig
from .exception_handlers import add_exception_handlers
from .middlewares import add_middlewares
from .recommendation_models import DSSMModel, LightFMModel, PopularModel, RangeModel, UserKnnPopModel
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

    # Initialize all model classes with logger
    models = {
        "userknn_pop": UserKnnPopModel(logger=app_logger),
        "lightfm_4f": LightFMModel(logger=app_logger),
        "model_range": RangeModel(logger=app_logger),
        "model_pop": PopularModel(logger=app_logger),
        "model_dssm": DSSMModel(logger=app_logger),
    }

    # Store models in app.state
    app.state.reco_models = models

    add_views(app)
    add_middlewares(app)
    add_exception_handlers(app)

    return app
