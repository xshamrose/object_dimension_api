from fastapi import APIRouter
from .measurement import router as measurement_router

api_router = APIRouter()
api_router.include_router(measurement_router)