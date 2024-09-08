import os
import redis
import dotenv
dotenv.load_dotenv()
client = redis.Redis.from_url(
    os.getenv("REDIS_URI"),
    decode_responses=True
)
