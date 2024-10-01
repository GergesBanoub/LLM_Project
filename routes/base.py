from fastapi import FastAPI,APIRouter

base = APIRouter()
@base.get("/")
def welcome ():
    return {"the web service is up and running"
    }