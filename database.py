"""
database.py — Async MongoDB client management via Motor.

Provides a lazy-initialized AsyncIOMotorClient and helpers to access
the clipmatch database and its collections.
"""

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from config import MONGO_URI, MONGO_DB_NAME

# Module-level client — initialized once on first access.
_client: AsyncIOMotorClient | None = None


def get_client() -> AsyncIOMotorClient:
    """Return (and lazily create) the shared Motor client."""
    global _client
    if _client is None:
        _client = AsyncIOMotorClient(MONGO_URI)
    return _client


def get_database() -> AsyncIOMotorDatabase:
    """Return the clipmatch Motor database handle."""
    return get_client()[MONGO_DB_NAME]


async def close_client() -> None:
    """Gracefully close the Motor client (call on app shutdown)."""
    global _client
    if _client is not None:
        _client.close()
        _client = None
