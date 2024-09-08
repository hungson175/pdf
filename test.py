from threading import Thread
from typing import Optional, Any

from langchain.chat_models.openai import ChatOpenAI
from langchain_core.outputs import LLMResult
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain, ConversationalRetrievalChain

from langchain.callbacks.base import BaseCallbackHandler
from dotenv import load_dotenv
from queue import Queue

from uuid import UUID

load_dotenv()


class StreamingHandler(BaseCallbackHandler):
    def __init__(self, queue):
        self.queue = queue

    def on_llm_new_token(self, token, **kwargs):
        self.queue.put(token)

    def on_llm_end(
            self,
            response: LLMResult,
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            **kwargs: Any,
    ) -> Any:
        self.queue.put(None)

    def on_llm_error(
            self,
            error: BaseException,
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            **kwargs: Any,
    ) -> Any:
        self.queue.put(None)


chat = ChatOpenAI(streaming=True)
prompt = ChatPromptTemplate.from_messages([
    ("human", "{content}")
])


# class StreamingLLMChain(LLMChain):
class StreamableChain:
    def stream(self, input):
        queue = Queue()
        handler = StreamingHandler(queue=queue)

        def task():
            self(input, callbacks=[handler])

        Thread(target=task).start()

        while True:
            token = queue.get()
            if token == None:
                return
            yield token


class StreamingLLMChain(StreamableChain, LLMChain):
    # Wow, the order is important: StreamableChain must be first to overwrite the stream() method
    pass


chain = StreamingLLMChain(llm=chat, prompt=prompt)

for output in chain.stream(input={"content": "Tell me a joke"}):
    print(output)
