"""
database.py — Async MongoDB client management via Motor.

Provides a lazy-initialized AsyncIOMotorClient and helpers to access
the clipmatch database and its collections.
"""

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
import pymongo
from config import MONGO_URI, MONGO_DB_NAME

# Module-level async client — initialized once on first access.
_client: AsyncIOMotorClient | None = None
# Module-level synchronous client — for CLI and background workers
_sync_client: pymongo.MongoClient | None = None


def get_client() -> AsyncIOMotorClient:
    """Return (and lazily create) the shared Motor client."""
    global _client
    if _client is None:
        _client = AsyncIOMotorClient(MONGO_URI)
    return _client


def get_database() -> AsyncIOMotorDatabase:
    """Return the clipmatch Motor database handle."""
    return get_client()[MONGO_DB_NAME]


def get_sync_client() -> pymongo.MongoClient:
    """Return (and lazily create) the shared synchronous PyMongo client."""
    global _sync_client
    if _sync_client is None:
        _sync_client = pymongo.MongoClient(
            MONGO_URI,
            serverSelectionTimeoutMS=1500,
            connectTimeoutMS=1500,
        )
    return _sync_client


def get_sync_database() -> pymongo.database.Database:
    """Return the synchronous clipmatch PyMongo database handle."""
    return get_sync_client()[MONGO_DB_NAME]


async def close_client() -> None:
    """Gracefully close the Motor client (call on app shutdown)."""
    global _client
    if _client is not None:
        _client.close()
        _client = None
