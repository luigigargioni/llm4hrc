from enum import Enum
import os

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.tools import BaseTool
from langchain_core.runnables import Runnable
from langchain_core.messages import BaseMessage
from langchain_core.language_models import LanguageModelInput
from langchain_groq import ChatGroq


class LlmProvider(str, Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"
    GROQ = "groq"


class OpenAiModel(str, Enum):
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4O = "gpt-4o"
    GPT_4_1 = "gpt-4.1"
    GPT_4_1_MINI = "gpt-4.1-mini"
    GPT_4_1_NANO = "gpt-4.1-nano"


class OllamaModel(str, Enum):
    LLAMA_3_1 = "llama3.1"
    LLAMA_3_3 = "llama3.3"


class GroqModel(str, Enum):
    LLAMA_3_1 = "llama-3.1-8b-instant"
    LLAMA_3_3 = "llama-3.3-70b-versatile"


TEMPERATURE = 0.5


def define_model(
    response_template: BaseTool | None = None,
) -> Runnable[LanguageModelInput, BaseMessage]:
    ollama_model = os.getenv("OLLAMA_MODEL") or OllamaModel.LLAMA_3_1.value
    openai_model = os.getenv("OPENAI_MODEL") or OpenAiModel.GPT_4O_MINI.value
    groq_model = os.getenv("GROQ_MODEL") or GroqModel.LLAMA_3_1.value
    model = None

    if os.getenv("LLM_PROVIDER") == LlmProvider.OPENAI.value:
        if response_template:
            model = ChatOpenAI(model=openai_model, temperature=TEMPERATURE).bind_tools(
                tools=[response_template],
                tool_choice="required",
                strict=True,
                parallel_tool_calls=False,
            )
        else:
            model = ChatOpenAI(model=openai_model, temperature=TEMPERATURE)
    elif os.getenv("LLM_PROVIDER") == LlmProvider.GROQ.value:
        if response_template:
            model = ChatGroq(model=groq_model, temperature=TEMPERATURE).bind_tools(
                [response_template]
            )
        else:
            model = ChatGroq(model=groq_model, temperature=TEMPERATURE)
    elif os.getenv("LLM_PROVIDER") == LlmProvider.OLLAMA.value:
        if response_template:
            model = ChatOllama(
                model=ollama_model, temperature=TEMPERATURE, format="json"
            ).bind_tools([response_template])
        else:
            model = ChatOllama(model=ollama_model, temperature=TEMPERATURE)
    else:
        raise ValueError("Invalid LLM_PROVIDER")

    return model
