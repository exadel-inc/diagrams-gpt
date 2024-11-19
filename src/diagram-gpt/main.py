from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.mermaid.api import api as mermaid_api
from app.plant_uml.api import api as plant_uml_api
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="Diagrams AI",
    version="1.0.0",
    docs_url="/swagger",
    swagger_ui_oauth2_redirect_url="/swagger/oauth2-redirect",
)


@app.exception_handler(RequestValidationError)
async def request_validation_error_handler(_: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"errors": [err.get("msg") for err in exc.errors() if err.get("msg")]},
    )


@app.exception_handler(ValueError)
async def validation_error_handler(_: Request, exc: ValueError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"errors": [str(exc)]},
    )


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*"
    ],  # for production, this should be limited to the frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(mermaid_api)
app.include_router(plant_uml_api)
