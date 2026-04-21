"""Initialize or refresh the SQLite database schema."""

from database import ensure_database_ready


def main():
    """Create tables and seed lookup data safely."""
    ensure_database_ready()
    print("Database setup complete.")


if __name__ == "__main__":
    main()
