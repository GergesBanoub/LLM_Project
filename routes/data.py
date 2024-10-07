from fastapi import FastAPI,APIRouter,Depends
from helpers.config import get_settings  

data_router = APIRouter(
 prefix="/api/v1/data",
 tags=["api_v1","data"],

)
@data_router.post("/upload")
def welcome (app_settings=Depends(get_settings)):
    app_name = app_settings.APP_NAME
    app_version = app_settings.APP_VERSION
    return {
        "app_name" : app_name ,
        "app_version": app_version
            
    }