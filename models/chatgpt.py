import json
import time
from typing import Generator

import openai

from models.base import ContextAwareModalStreamingHandler


class ChatGPTStreamingHandler(ContextAwareModalStreamingHandler):

    def __init__(self, api_key: str, model: str = "gpt-5-nano"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.tools = self._get_weather_tools()
        self.system_message = self._get_system_message()

    def stream_chat(self, user_message: str) -> Generator[str, None, None]:

        # Log user message
        self.set_model_name(self.model)
        self.set_user_message(user_message)

        start_time = time.time()

        try:
            # Check if tools are needed with system message
            messages = [{"role": "system", "content": self.system_message}, {"role": "user", "content": user_message}]

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools,
                tool_choice="auto",
                stream=False,  # Non-streaming to check for tools
            )

            message = response.choices[0].message
            messages.append(message)

            if message.tool_calls:
                yield from self._handle_with_tools(message.tool_calls, messages)
            else:
                yield self._format_sse({"type": "text_start"})
                yield self._format_sse({"type": "text_delta", "delta": message.content})

        except Exception as e:
            yield self._format_sse({"type": "error", "message": f"ChatGPT error: {str(e)}"})

        finally:
            # Always send completion
            total_time = round((time.time() - start_time) * 1000)
            yield self._format_sse({"type": "done", "total_time": total_time, "model": self.model})

    def _handle_with_tools(self, tool_calls, messages):

        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            # Notify frontend about tool execution
            yield self._format_sse({"type": "tool_call", "function_name": function_name, "arguments": function_args})

            # Execute the tool
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

            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(tool_result)})

        # Stream commentary about the weather data
        yield from self._stream_weather_commentary(messages)

    def _stream_weather_commentary(self, messages):
        """Stream commentary about weather data without repeating the data"""
        try:
            commentary_prompt = self._build_commentary_prompt()

            messages.append({"role": "user", "content": commentary_prompt})

            # Signal that commentary streaming is starting
            yield self._format_sse({"type": "text_start"})

            stream = self.client.chat.completions.create(model=self.model, messages=messages, stream=True)

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield self._format_sse({"type": "text_delta", "delta": chunk.choices[0].delta.content})

        except Exception as e:
            yield self._format_sse({"type": "error", "message": f"Commentary streaming error: {str(e)}"})
