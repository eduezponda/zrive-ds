from fastapi import APIRouter, Response

from prometheus_client import core, exposition


router = APIRouter()


@router.get("/metrics")
def metrics():
    stats = exposition.generate_latest(core.REGISTRY)
    return Response(
        content=stats,
        headers={"Content-Type": "text/plain; version=0.0.4; charset=utf-8"}
    )
