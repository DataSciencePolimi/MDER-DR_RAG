"""Orchestrator module for handling live user messages."""
from langchain_core.messages import AIMessage, HumanMessage

from .abstract_orchestrator import AbstractOrchestrator
from .guru import Guru


class LiveOrchestrator(AbstractOrchestrator):
    """
    Orchestrator for live chat.
    """

    def __init__(self, provider: str, model: str, embedding: str, temperature: float,
                 language: str, answer_length: str, region: str, use_knowledge: bool = True) -> None:
        """Initialize the LiveOrchestrator with the model name, provider, and API key."""
        self.guru = Guru(
            provider=provider,
            model=model,
            embedding=embedding,
            language=language,
            temperature=temperature,
            answer_length=answer_length,
            region=region,
            use_knowledge=use_knowledge
        )
       
    def load_past_messages(self, messages: list) -> None:
        """
        Load past messages into the orchestrator.
        """
        typed_list = []
        for m in messages:
            if m["role"] == "user":
                typed_list.append(HumanMessage(content=m["content"]))
            elif m["role"] == "assistant":
                typed_list.append(AIMessage(content=m["content"]))

        self.guru.load_past_messages(typed_list)

    def user_message(self, message: str):# -> str:
        """
        Process a user message and return a response.
        """
        #return self.guru.user_message(message)
        yield from self.guru.user_message_stream(message)
