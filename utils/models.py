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
    OPENROUTER = "openrouter"


# OpenRouter speaks the OpenAI wire protocol, so ChatOpenAI drives it with a
# different base URL and key; no extra dependency is needed.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterModel(str, Enum):
    LLAMA_3_3_70B = "meta-llama/llama-3.3-70b-instruct"
    LLAMA_3_1_70B = "meta-llama/llama-3.1-70b-instruct"
    GPT_OSS_120B = "openai/gpt-oss-120b"
    QWEN_2_5_72B = "qwen/qwen-2.5-72b-instruct"


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


#TEMPERATURE = 0.5
#per una questione di token provo ad abbassare la temperature, 
# vediamo se migliora la coerenza delle risposte e se riesce a mantenere meglio il contesto, anche se potrebbe diventare più ripetitivo
TEMPERATURE = 0.3


def define_model(
    response_template: BaseTool | None = None,
    temperature: float | None = None,
    model: str | None = None,
    provider: str | None = None,
) -> Runnable[LanguageModelInput, BaseMessage]:
    """Build the configured chat model.

    ``temperature`` overrides the global default for callers that need a
    reproducible answer rather than a varied one (the Action Canonicalizer
    maps a name to a fixed vocabulary, so sampling only adds noise).

    ``model`` and ``provider`` let a single node run on a different model
    from the rest of the graph. The nodes do not all ask the same thing of
    a model: canonicalization picks one label out of a closed vocabulary,
    while task synthesis generates a large structured object. A small
    open-weight model handles the first perfectly and fails the second, so
    the choice is made per node rather than once for the whole system.
    Both default to the global configuration when omitted.
    """

    ollama_model = model or os.getenv("OLLAMA_MODEL") or OllamaModel.LLAMA_3_1.value
    openai_model = model or os.getenv("OPENAI_MODEL") or OpenAiModel.GPT_4O_MINI.value
    groq_model = model or os.getenv("GROQ_MODEL") or GroqModel.LLAMA_3_1.value

    provider = provider or os.getenv("LLM_PROVIDER")

    effective_temperature = (
        TEMPERATURE if temperature is None else temperature
    )

    #print(f"creating model with provider {provider}...")

    # OPENAI
    if provider == LlmProvider.OPENAI.value:

        model = ChatOpenAI(
            model=openai_model,
            temperature=effective_temperature,
        )

        if response_template:
            model = model.bind_tools(
                tools=[response_template],
                tool_choice="required",
                strict=True,
                parallel_tool_calls=False,
            )

        return model

    # OPENROUTER
    elif provider == LlmProvider.OPENROUTER.value:

        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is not set; it is required when "
                "LLM_PROVIDER=openrouter (the OpenAI key is not a "
                "substitute, the endpoints are different)"
            )

        openrouter_model = (
            model
            or os.getenv("OPENROUTER_MODEL")
            or OpenRouterModel.LLAMA_3_3_70B.value
        )

        # OpenRouter uses these headers for attribution on its dashboard;
        # both are optional and simply omitted when unset.
        headers = {}
        if os.getenv("OPENROUTER_SITE_URL"):
            headers["HTTP-Referer"] = os.getenv("OPENROUTER_SITE_URL", "")
        if os.getenv("OPENROUTER_APP_NAME"):
            headers["X-Title"] = os.getenv("OPENROUTER_APP_NAME", "")

        # Reasoning models spend most of their output budget on hidden
        # chain-of-thought. Mapping a name onto a closed vocabulary, or
        # filling a fixed schema, does not need it, and it dominates both
        # latency and output cost. OpenRouter takes a `reasoning` field;
        # "low" is the least these models accept, they cannot switch it off.
        extra_body = {}
        reasoning_effort = os.getenv("OPENROUTER_REASONING_EFFORT")
        if reasoning_effort:
            extra_body["reasoning"] = {"effort": reasoning_effort}

        # OpenRouter load-balances one model across many providers, which
        # for gpt-oss-120b run it at anything from fp4 to bf16 on different
        # hardware. Numerics that differ between providers make the same
        # prompt answer differently even at temperature 0, and make latency
        # swing. Pinning the provider, or at least the precision, is what
        # buys reproducibility; allow_fallbacks stays off so a request
        # fails rather than silently landing somewhere else.
        routing = {}
        order = os.getenv("OPENROUTER_PROVIDER_ORDER")
        if order:
            routing["order"] = [p.strip() for p in order.split(",") if p.strip()]
            routing["allow_fallbacks"] = (
                os.getenv("OPENROUTER_ALLOW_FALLBACKS", "false").lower() == "true"
            )
        quantizations = os.getenv("OPENROUTER_QUANTIZATIONS")
        if quantizations:
            routing["quantizations"] = [
                q.strip() for q in quantizations.split(",") if q.strip()
            ]
        if routing:
            extra_body["provider"] = routing

        # Same seed plus same provider is as close to a repeatable answer
        # as a hosted model gets; providers that ignore it simply sample.
        seed = os.getenv("OPENROUTER_SEED")
        if seed:
            extra_body["seed"] = int(seed)

        model = ChatOpenAI(
            model=openrouter_model,
            temperature=effective_temperature,
            base_url=(
                os.getenv("OPENROUTER_BASE_URL")
                or os.getenv("OPENROUTER_URL")
                or OPENROUTER_BASE_URL
            ),
            api_key=api_key,
            default_headers=headers or None,
            extra_body=extra_body or None,
        )

        if response_template:
            # No strict=True here: structured tool calling is an OpenAI
            # feature, and open-weight models served through OpenRouter
            # reject it. The two nodes that bind a tool already fall back
            # to parsing raw JSON when no tool call comes back.
            model = model.bind_tools(
                tools=[response_template],
                tool_choice="required",
            )

        return model

    # GROQ
    elif provider == LlmProvider.GROQ.value:

        model = ChatGroq(
            model=groq_model,
            temperature=effective_temperature,
        )

        if response_template:
            model = model.bind_tools(
                [response_template]
            )

        return model

    # OLLAMA
    elif provider == LlmProvider.OLLAMA.value:

        return ChatOllama(
            model=ollama_model,
            temperature=effective_temperature,
            format="json"
        )

    else:
        raise ValueError("Invalid LLM_PROVIDER")