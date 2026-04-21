"""Minimal Flask entry point."""

from app import app
from database import ensure_database_ready


if __name__ == "__main__":
    ensure_database_ready()
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
