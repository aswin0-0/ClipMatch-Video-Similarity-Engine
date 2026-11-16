
import asyncio
from database import get_database, close_client

async def test_connection():
    try:
        db = get_database()

        await db.command("ping")

        print("MongoDB connection successful!")
        print(f"Database: {db.name}")

    except Exception as e:
        print("MongoDB connection failed!")
        print(e)

    finally:
        await close_client()

asyncio.run(test_connection())
