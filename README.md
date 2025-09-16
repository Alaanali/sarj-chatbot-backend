# Weather Chatbot with Automated Evaluation System

A production-ready weather chatbot system with streaming responses, automated quality evaluation, and comprehensive analytics. The system enforces strict scope control to handle only weather-related queries while providing real-time weather data through multiple AI models.

## Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │───▶│   Flask API      │───▶│  AI Models      │
│   (SSE Client)  │    │  (Streaming)     │    │  GPT/Gemini     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │   Context        │
                       │   Manager        │
                       └──────────────────┘
                                │
                                ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │   SQLite DB      │───▶│   Automated     │
                       │   (Conversations)│    │   Evaluator     │
                       └──────────────────┘    └─────────────────┘
```

## Core Components

### 1. **Context Management System**
- **ConversationState**: Dataclass-based state management with contextvars
- **Thread-safe**: Per-request context isolation using Python's contextvars
- **Lifecycle Tracking**: Complete conversation flow from user input to evaluation
- **Tool Call Management**: Tracks pending and resolved function calls

```python
@dataclass
class ConversationState:
    session_id: str
    conversation_id: Optional[int]
    response_content: str
    tool_calls: List[dict]
    pending_tool_call: Optional[Dict[str, Any]]
```

### 2. **Database Schema**
**Tables:**
- `conversations`: Session tracking with user metadata
- `messages`: User/assistant messages with performance metrics
- `tool_calls`: Function call execution logs with timing
- `evaluations`: LLM-based quality assessments

**Key Features:**
- WAL mode for concurrent access
- Relationship mapping with cascade deletes
- Computed properties for analytics aggregation

### 3. **AI Model Handlers**
**Base Architecture:**
```python
class ContextAwareModalStreamingHandler(ModalStreamingHandler):
    def stream_chat(self, user_message: str) -> Generator[str, None, None]
    def _execute_tool(self, function_name: str, function_args: dict) -> dict
    def _format_sse(self, data: dict) -> str
```

**Supported Models:**
- **GPT-5 Nano**: OpenAI's fast model with tool calling
- **Gemini 2.0 Flash Lite**: Google's efficient model with function declarations

### 4. **Weather Integration**
**Tools Available:**
- `get_current_weather(city, units)`: Real-time weather data
- `get_weather_forecast(city, days)`: Up to 5-day forecasts

**Data Processing:**
- OpenWeatherMap API integration with error handling
- Temperature unit conversion (Celsius/Fahrenheit)
- Forecast aggregation with daily min/max calculations

### 5. **Automated Evaluation Pipeline**
**LLMEvaluator Class:**
```python
def evaluate_message(self, message_id: int) -> Optional[Dict[str, Any]]
def batch_evaluate_unevaluated(self, limit: int = 50) -> Dict[str, Any]
```

**Evaluation Dimensions:**
- Helpfulness (1-10): Response utility and completeness
- Correctness (1-10): Factual accuracy and logical soundness
- Politeness (1-10): Professional tone and courtesy
- Accuracy (1-10): Weather data precision and clarity
- Scope Adherence (1-10): Weather-only focus compliance

## API Endpoints

### Core Endpoints
```bash
POST /api/chat/stream          # Streaming chat with SSE
GET  /api/dashboard/stats      # System performance metrics
GET  /api/conversations        # Recent conversation list
GET  /api/conversation/<id>    # Detailed conversation view
POST /api/evaluation/run       # Trigger batch evaluation
GET  /api/evaluation/status    # Evaluation progress
GET  /api/analytics/trends     # Performance trends
```

### Server-Sent Events Format
```javascript
data: {"type": "text_start"}
data: {"type": "text_delta", "delta": "Weather in"}
data: {"type": "tool_call", "function_name": "get_current_weather", "arguments": {...}}
data: {"type": "weather_data", "data": {...}, "execution_time": 250}
data: {"type": "done", "total_time": 1500, "model": "gpt-5-nano"}
```

## Installation & Setup

### Prerequisites
```bash
Python 3.9+
OpenWeatherMap API Key
OpenAI API Key (for GPT models)
Google AI Studio API Key (for Gemini models)
```

### Environment Configuration
```bash
# .env file
OPENWEATHERMAP_API_KEY=your_weather_api_key
OPENAI_API_KEY=your_openai_api_key
GOOGLE_API_KEY=your_google_api_key
```

## Technical Features

### Conversation Context Management
- **Thread-safe State**: contextvars implementation for concurrent requests
- **Automatic Lifecycle**: Start → User Message → Assistant Response → Tool Calls → Finalization
- **Error Handling**: Comprehensive error tracking with conversation state preservation

### Streaming Architecture
- **Server-Sent Events**: Real-time response streaming to frontend
- **Tool Call Integration**: Seamless weather data insertion during response generation
- **Commentary Generation**: Post-tool-call AI commentary without data repetition

### Quality Assurance
- **Scope Enforcement**: System prompt strictly limits responses to weather queries
- **Automatic Evaluation**: Background LLM evaluation of all assistant responses
- **Performance Tracking**: Response times, tool success rates, evaluation scores

### Analytics & Monitoring
- **Real-time Metrics**: Dashboard with live performance indicators
- **Trend Analysis**: Daily performance comparisons between AI models
- **Conversation Insights**: Complete audit trail of all interactions


### Code Structure
```
├── context/           # Conversation state management
├── database/          # SQLAlchemy models and operations
├── evaluators/        # LLM-based evaluation system
├── models/            # AI model handlers (GPT/Gemini)
├── tools/             # Weather API integration
└── app.py            # Flask application entry point
```