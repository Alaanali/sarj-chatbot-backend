import contextvars
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from database.models import db_manager


@dataclass
class ConversationState:
    # Conversation identifiers
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: Optional[int] = None

    # User info
    user_ip: Optional[str] = None
    user_agent: Optional[str] = None

    # Model info
    model_name: Optional[str] = None

    # Messages
    current_user_message_id: Optional[int] = None
    current_assistant_message_id: Optional[int] = None

    # Response tracking
    response_content: str = ""
    response_start_time: Optional[float] = None

    # Tool calls
    tool_calls: List[dict] = field(default_factory=list)
    pending_tool_call: Optional[Dict[str, Any]] = None

    # Errors
    error_occurred: bool = False
    error_message: Optional[str] = None

    # Metadata
    created_at: float = field(default_factory=time.time)


# ContextVar holds a ConversationState per request/thread/task
conversation_context: contextvars.ContextVar[Optional[ConversationState]] = contextvars.ContextVar(
    "conversation_context", default=None
)


class ContextManager:
    """Manages conversation context for the current request"""

    @staticmethod
    def _get_state() -> ConversationState:
        state = conversation_context.get()
        if state is None:
            raise Exception("Conversation state not initialized")
        return state

    @staticmethod
    def _set_state(state: ConversationState):
        conversation_context.set(state)

    # --- Conversation lifecycle ---

    @staticmethod
    def start_conversation(session_id=None, user_ip=None, user_agent=None, model_name=None) -> str:
        if not session_id:
            session_id = str(uuid.uuid4())

        conversation_id = db_manager.get_or_create_conversation(session_id, user_ip, user_agent)

        state = ConversationState(
            session_id=session_id,
            conversation_id=conversation_id,
            model_name=model_name,
        )
        ContextManager._set_state(state)
        return session_id

    # --- User message ---

    @staticmethod
    def set_user_message(content: str) -> int:
        state = ContextManager._get_state()
        if not state.conversation_id:
            ContextManager.start_conversation()
            state = ContextManager._get_state()

        state.current_user_message_id = db_manager.create_message(
            conversation_id=state.conversation_id,
            role="user",
            content=content,
        )
        ContextManager._set_state(state)
        return state.current_user_message_id

    # --- Assistant response ---

    @staticmethod
    def start_assistant_response():
        state = ContextManager._get_state()
        state.response_content = ""
        state.response_start_time = time.time()
        ContextManager._set_state(state)
        pass

    @staticmethod
    def append_response(delta: str):
        state = ContextManager._get_state()
        state.response_content += delta
        ContextManager._set_state(state)

    # --- Tool calls ---

    @staticmethod
    def set_pending_tool_call(function_name: str, arguments: dict):
        state = ContextManager._get_state()
        state.pending_tool_call = {"function_name": function_name, "arguments": arguments}
        ContextManager._set_state(state)

    @staticmethod
    def resolve_pending_tool_call(result: dict, execution_time_ms: int, error_message: Optional[str] = None):
        state = ContextManager._get_state()
        if not state.pending_tool_call:
            return
        state.tool_calls.append(
            {
                "function_name": state.pending_tool_call["function_name"],
                "arguments": state.pending_tool_call["arguments"],
                "result": result,
                "execution_time_ms": execution_time_ms,
                "success": error_message is None,
                "error_message": error_message,
            }
        )
        state.pending_tool_call = None
        print("adding call", state.tool_calls)
        ContextManager._set_state(state)

    # --- Finalization ---

    @staticmethod
    def finalize_assistant_response(error_occurred: bool = False, error_message: Optional[str] = None) -> Optional[int]:
        state = ContextManager._get_state()
        if not state.conversation_id:
            return None

        response_time_ms = round((time.time() - state.response_start_time) * 1000)

        state.current_assistant_message_id = db_manager.create_message(
            conversation_id=state.conversation_id,
            role="assistant",
            content=state.response_content,
            model_name=state.model_name,
            response_time_ms=response_time_ms,
            error_occurred=error_occurred,
            error_message=error_message,
        )

        # Save tool calls
        for tool_call in state.tool_calls:
            db_manager.create_tool_call(message_id=state.current_assistant_message_id, **tool_call)

        state.tool_calls = []
        ContextManager._set_state(state)
        return state.current_assistant_message_id
