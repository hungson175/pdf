from app.chat.chains.retrieval import StreamingConversationalRetrievalChain
from app.chat.llms import llm_map
from app.chat.memories import memory_map
from app.chat.models import ChatArgs
from langchain.chat_models.openai import ChatOpenAI
from app.chat.vector_store import retrieval_map
from app.chat.score import random_component_by_score
from langfuse.model import CreateTrace
from app.web.api import (
    set_conversation_components,
    get_conversation_components
)
from app.chat.tracing.langfuse import langfuse

# disable the deprecated warning
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

def select_component(component_type, component_map, chat_args):
    components = get_conversation_components(chat_args.conversation_id)
    prev_component = components[component_type]
    component = None
    if prev_component:
        # Not the first message use the same component
        build_component = component_map[prev_component]
        component = build_component(chat_args)
        return prev_component, component
    else:
        # First message
        # Create a new component
        component_name = random_component_by_score(component_type, component_map)
        build_component = component_map[component_name]
        component = build_component(chat_args)
        return component_name, component


def build_chat(chat_args: ChatArgs):
    """
    :param chat_args: ChatArgs object containing
        conversation_id, pdf_id, metadata, and streaming flag.

    :return: A chain

    Example Usage:

        chain = build_chat(chat_args)
    """
    retriever_name, retriever = select_component(
        "retriever",
        retrieval_map,
        chat_args
    )

    llm_name, llm = select_component(
        "llm",
        llm_map,
        chat_args
    )

    memory_name, memory = select_component(
        "memory",
        memory_map,
        chat_args
    )

    print(f"Running with components: retriever={retriever_name}, llm={llm_name}, memory={memory_name}")
    set_conversation_components(
        conversation_id=chat_args.conversation_id,
        retriever=retriever_name,
        llm=llm_name,
        memory=memory_name
    )

    condense_question_llm = ChatOpenAI(model="gpt-4o", streaming=False)

    trace = langfuse.trace(
        CreateTrace(
            id=chat_args.conversation_id,
            metadata=chat_args.metadata,
        )
    )

    return StreamingConversationalRetrievalChain.from_llm(
        llm=llm,
        condense_question_llm=condense_question_llm,
        memory=memory,
        retriever=retriever,
        metadata=chat_args.metadata,
    )
