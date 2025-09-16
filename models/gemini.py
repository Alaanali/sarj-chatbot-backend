import time
from typing import Generator

import google.generativeai as genai

from models.base import ContextAwareModalStreamingHandler


class GeminiStreamingHandler(ContextAwareModalStreamingHandler):
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash-lite"):
        genai.configure(api_key=api_key)
        self.model_name = model
        self.model = genai.GenerativeModel(
            model_name=model, tools=self._get_weather_tools(), system_instruction=self._get_system_message()
        )

    def stream_chat(self, user_message: str) -> Generator[str, None, None]:

        self.set_model_name(self.model_name)
        self.set_user_message(user_message)

        start_time = time.time()

        try:
            chat = self.model.start_chat()
            response = chat.send_message(user_message, stream=False)

            # Check if any part is a function call
            has_function_calls = any(
                hasattr(part, "function_call") and part.function_call for part in response.candidates[0].content.parts
            )

            if has_function_calls:
                yield from self._handle_with_tools(chat, response)
            else:
                text_content = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, "text"))
                yield self._format_sse({"type": "text_start"})
                yield self._format_sse({"type": "text_delta", "delta": text_content})

        except Exception as e:
            yield self._format_sse({"type": "error", "message": f"Gemini error: {str(e)}"})

        finally:
            # Always send completion
            total_time = round((time.time() - start_time) * 1000)
            yield self._format_sse({"type": "done", "total_time": total_time, "model": self.model_name})

    def _handle_with_tools(self, chat, initial_response):

        function_calls = []
        for part in initial_response.candidates[0].content.parts:
            if hasattr(part, "function_call") and part.function_call:
                function_calls.append(part.function_call)

        # Execute each function call
        function_responses = []
        for function_call in function_calls:
            function_name = function_call.name
            function_args = dict(function_call.args)

            # Notify frontend about tool execution
            yield self._format_sse({"type": "tool_call", "function_name": function_name, "arguments": function_args})

            tool_start = time.time()
            tool_result = self._execute_tool(function_name, function_args)
            tool_time = round((time.time() - tool_start) * 1000)

            # Send weather data to frontend for visual display
            yield self._format_sse(
                {
                    "type": "weather_data",
                    "data": tool_result,
                    "execution_time": tool_time,
                    "city": function_args.get("city", "Unknown"),
                }
            )

            function_response = genai.protos.Part(
                function_response=genai.protos.FunctionResponse(name=function_name, response={"result": tool_result})
            )
            function_responses.append(function_response)

        if function_responses:
            yield from self._stream_weather_commentary(chat, function_responses)

    def _stream_weather_commentary(self, chat, function_responses):

        try:
            commentary_message = genai.protos.Content(
                parts=function_responses + [genai.protos.Part(text=self._build_commentary_prompt())]
            )

            # Signal that commentary streaming is starting
            yield self._format_sse({"type": "text_start"})

            response = chat.send_message(commentary_message, stream=True)

            for chunk in response:
                if chunk.text:
                    yield self._format_sse({"type": "text_delta", "delta": chunk.text})

        except Exception as e:
            yield self._format_sse({"type": "error", "message": f"Commentary streaming error: {str(e)}"})

    def _get_weather_tools(self) -> list:
        """Convert weather tools to Gemini format"""
        return [
            genai.protos.Tool(
                function_declarations=[
                    genai.protos.FunctionDeclaration(
                        name="get_current_weather",
                        description="Get current weather information for a specific city",
                        parameters=genai.protos.Schema(
                            type=genai.protos.Type.OBJECT,
                            properties={
                                "city": genai.protos.Schema(
                                    type=genai.protos.Type.STRING,
                                    description="The city name to get weather for (e.g., 'London', 'New York')",
                                ),
                                "units": genai.protos.Schema(
                                    type=genai.protos.Type.STRING,
                                    enum=["celsius", "fahrenheit"],
                                    description="Temperature units to use",
                                ),
                            },
                            required=["city"],
                        ),
                    ),
                    genai.protos.FunctionDeclaration(
                        name="get_weather_forecast",
                        description="Get weather forecast for a specific city",
                        parameters=genai.protos.Schema(
                            type=genai.protos.Type.OBJECT,
                            properties={
                                "city": genai.protos.Schema(
                                    type=genai.protos.Type.STRING, description="The city name to get forecast for"
                                ),
                                "days": genai.protos.Schema(
                                    type=genai.protos.Type.INTEGER, description="Number of days for forecast (1-5)"
                                ),
                            },
                            required=["city"],
                        ),
                    ),
                ]
            )
        ]
