import os

from models.base import ModalStreamingHandler
from models.chatgpt import ChatGPTStreamingHandler
from models.gemini import GeminiStreamingHandler


def get_model(model_name: str) -> ModalStreamingHandler:
    print(model_name)
    if model_name in ["gpt-5-nano"]:
        api_key = os.getenv("OPENAI_API_KEY")
        return ChatGPTStreamingHandler(api_key, model_name)

    elif model_name in ["gemini-2.0-flash-lite"]:
        api_key = os.getenv("GEMINI_API_KEY")
        return GeminiStreamingHandler(api_key, model_name)
