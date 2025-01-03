#app/core/config.py

from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Object Dimension Detection API"
    MODEL_PATH: str = "models/midas_model.pt"
    UPLOAD_FOLDER: str = "uploads"
    
    # Database settings
    DATABASE_URL: str = "sqlite:///./objects.db"
    
    class Config:
        env_file = ".env"

settings = Settings()