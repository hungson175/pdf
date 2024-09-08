import os

import dotenv
from langfuse.client import Langfuse
dotenv.load_dotenv()
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST_URL"),
)