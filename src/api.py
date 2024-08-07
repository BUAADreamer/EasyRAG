# -*- coding: UTF-8 -*-
import uvicorn
from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from easyrag.pipeline.pipeline import EasyRAGPipeline
from easyrag.utils import get_yaml_data


class QueryRequest(BaseModel):
    query: str = ""
    document: str = ""


class QueryResponse(BaseModel):
    answer: str = ""
    contexts: list[str] = []


def create_app() -> FastAPI:
    app = FastAPI()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return app


config_path = "configs/easyrag.yaml"
config = get_yaml_data(config_path)
easyrag = EasyRAGPipeline(config)

app = create_app()


@app.get("/test")
def test():
    return "hello rag"


@app.post("/v1/rag", status_code=status.HTTP_200_OK)
async def rag(request: QueryRequest):
    # query对象: {"query":"Daisyseed安装软件从哪里获取", "document":"director"}
    query = {"query": request.query, "document": request.document}
    res = await easyrag.run(
        query
    )
    result = {
        "answer": res["answer"],
        "contexts": res["contexts"],
    }
    return result
