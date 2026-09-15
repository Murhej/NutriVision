"""
NutriVision API package.

Backward-compatible entrypoint:
    python -m src.api           # starts uvicorn on :8000
    from src.api import app     # use app in tests
"""

from src.api.app import app

__all__ = ["app"]


def _run_server() -> None:
    import uvicorn
    print("\n" + "=" * 60)
    print("NutriVision API Server")
    print("=" * 60)
    print("Docs:     http://localhost:8000/docs")
    print("Frontend: http://localhost:8000/static/index.html")
    print("Press Ctrl+C to stop")
    print("=" * 60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    _run_server()
