from __future__ import annotations

from mangum import Mangum

from app.tempmain import app

handler = Mangum(app, lifespan="off")

__all__ = ["handler"]
