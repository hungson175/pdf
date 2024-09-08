from functools import partial
from .chatopenai import build_llm
llm_map = {
    "gpt-4o": partial(build_llm, model_name="gpt-4o"),
    "gpt-3.5-turbo": partial(build_llm, model_name="gpt-3.5-turbo"),
}