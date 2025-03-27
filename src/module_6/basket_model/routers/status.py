from fastapi import APIRouter
from fastapi.responses import Response

router = APIRouter()

@router.get("/status")
def status():
    return Response(status_code=200)
