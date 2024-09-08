from langchain.chat_models.openai import ChatOpenAI
# disable the deprecated warning
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


def build_llm(chat_args, model_name):
    return ChatOpenAI(model=model_name, streaming=chat_args.streaming)
