from typing import Any, Dict

from langchain_core.prompt_values import StringPromptValue
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from ollama import Client as OllamaClient


class MixtralLLM(Runnable):
    def __init__(self, system_prompt: str = "You are a helpful assistant."):

        self.client = OllamaClient()
        self.system_prompt = system_prompt
        self.model = "mixtral"

    def invoke(self, input: StringPromptValue, config: Dict[str, Any] = None) -> str:
        response = self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": input.text},
            ],
        )
        return response["message"]["content"]


class OpenAILLM(Runnable):
    def __init__(
        self,
        system_prompt: str = "You are a helpful assistant.",
        model: str = "gpt-3.5-turbo",
    ):
        self.system_prompt = system_prompt
        self.model = model
        self.llm = ChatOpenAI(model=self.model, temperature=0)

    def invoke(self, input: StringPromptValue, config: Dict[str, Any] = None) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": input.text},
        ]
        response = self.llm.invoke(messages)
        return response.content
