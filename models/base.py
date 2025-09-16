import json
from typing import Generator, Protocol

from tools.weather import get_current_weather, get_weather_forecast


class ModalStreamingHandler(Protocol):

    def stream_chat(self, user_message: str) -> Generator[str, None, None]: ...

    def _get_system_message(self) -> str:
        """System message to restrict model to weather-related queries only"""
        return """
            You are a specialized weather assistant. Your ONLY function is to help users with weather-related queries.

            WEATHER QUERIES (you should handle):
            - Current weather conditions for cities
            - Weather forecasts
            - Travel weather advice
            - Weather comparisons between cities
            - Seasonal weather questions
            - Weather-appropriate clothing or activity suggestions

            NON-WEATHER QUERIES (you should politely decline):
            - Math problems
            - Programming questions
            - General knowledge questions
            - Any topic not related to weather

            For non-weather queries, politely respond: "I'm a specialized weather assistant. I can help you with weather forecasts, current conditions, and weather-related advice. What weather information can I provide for you today?"

            When handling weather queries:
            1. Use available tools to get current data
            2. If weather data is displayed in visual cards, provide commentary without repeating the specific numbers
            3. Focus on practical advice and insights rather than restating displayed data

            Be helpful, concise, and weather-focused in all interactions.
        """.strip()

    def _get_weather_tools(self) -> list:
        """Define available weather tools"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "Get current weather information for a specific city",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "The city name to get weather for (e.g., 'London', 'New York')",
                            },
                            "units": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                                "description": "Temperature units to use",
                                "default": "celsius",
                            },
                        },
                        "required": ["city"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_weather_forecast",
                    "description": "Get weather forecast for a specific city",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "The city name to get forecast for"},
                            "days": {
                                "type": "integer",
                                "description": "Number of days for forecast (1-5)",
                                "default": 5,
                                "minimum": 1,
                                "maximum": 5,
                            },
                        },
                        "required": ["city"],
                    },
                },
            },
        ]

    def _format_sse(self, data: dict) -> str:
        """Format data as Server-Sent Event"""
        return f"data: {json.dumps(data)}\n\n"

    def _build_commentary_prompt(self) -> str:
        """Build prompt for weather commentary that avoids data repetition"""
        return f"""
            The weather data has been displayed to the user in visual cards with all the specific details (temperature, humidity, wind, etc.).
            Please provide helpful short commentary about this weather WITHOUT repeating any of the numerical data or specific conditions that are already shown in the cards.
            Be conversational and helpful, but avoid restating the specific weather details that are already visually displayed.
        """.strip()

    def _execute_tool(self, function_name: str, function_args: dict) -> dict:
        """Execute weather tool functions"""
        try:
            if function_name == "get_current_weather":
                return get_current_weather(**function_args)
            elif function_name == "get_weather_forecast":
                return get_weather_forecast(**function_args)
            else:
                return {
                    "error": f"Unknown function: {function_name}",
                    "available_functions": ["get_current_weather", "get_weather_forecast"],
                }
        except Exception as e:
            return {"error": f"Tool execution failed: {str(e)}", "function": function_name, "arguments": function_args}
