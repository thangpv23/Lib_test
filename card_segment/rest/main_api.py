import os
import pathlib
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# CWD = pathlib.Path(__file__).resolve().parents[3]
# sys.path.append(os.path.abspath(CWD))

import uvicorn

from loguru import logger

from fastapi import FastAPI

from api import card_segment_api 

app = FastAPI(
    title="Yolact Card Segment API", description="Yolact  Segmentation "
)


def Config_routing():
    app.include_router(card_segment_api.router)



if __name__ == "__main__":
    Config_routing()

    uvicorn.run(app, host="0.0.0.0", port=8010, reload=False)

    print("Done")
