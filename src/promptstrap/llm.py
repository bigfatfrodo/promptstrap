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

    def invoke(self, input: StringPromptValue, config: Dict[str, Any] = {}) -> str:
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
        model: str = "gpt-4.1",
        tools: list = [],
    ):
        self.system_prompt = system_prompt
        self.model = model
        self.llm = ChatOpenAI(model=self.model, temperature=0.3)
        self.tools = {t.name: t for t in tools}
        self.llm = self.llm.bind_tools(tools)

    def invoke(self, input: StringPromptValue, config: Dict[str, Any] = {}) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": input.text},
        ]
        response = self.llm.invoke(messages)
        for tool in response.tool_calls or []:
            if tool["name"] in self.tools:
                tool_response = self.tools[tool["name"]].invoke(tool["args"])
                print(f"Tool Response: {tool_response}\n")

        return response.content
