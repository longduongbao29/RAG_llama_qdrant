import uvicorn
from fastapi import FastAPI
from rag.routers.api import router

app = FastAPI()


app.include_router(router)


if __name__ == "__main__":
    uvicorn.run("main:app", port=1234, reload=True)
