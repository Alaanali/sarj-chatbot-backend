import json
import os
import threading
from datetime import datetime, timedelta

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, request
from flask_cors import CORS

from context.manager import ContextManager
from database.models import db_manager
from evaluators.chatgpt import LLMEvaluator
from models import get_model

load_dotenv()

app = Flask(__name__)
CORS(app)

# Initialize evaluator
evaluator = LLMEvaluator(api_key=os.getenv("OPENAI_API_KEY"))

# Session storage (in production, we can use Redis)
sessions = {}


@app.route("/api/health")
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "models_available": ["gpt-5-nano", "gemini-2.0-flash-lite"]})


@app.route("/api/chat/stream", methods=["POST"])
def stream_chat():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request body must be JSON"}), 400

        message = data.get("message", "").strip()
        model = data.get("model", "gpt-5-nano")
        session_id = data.get("session_id")

        if not message:
            return jsonify({"error": "Message is required"}), 400

        if model not in ["gpt-5-nano", "gemini-2.0-flash-lite"]:
            return jsonify({"error": f"Unsupported model: {model}"}), 400

        # Get client info
        user_ip = request.remote_addr
        user_agent = request.headers.get("User-Agent", "")

        handler = get_model(model)
        if not handler:
            return jsonify({"error": f"Unsupported model: {model}"}), 400

        # Initialize conversation context
        ContextManager.start_conversation(session_id=session_id, user_ip=user_ip, user_agent=user_agent, model_name=model)

        def generate():
            try:
                for chunk in handler.stream_chat(message):
                    yield chunk
            except Exception as e:
                error_data = {"type": "error", "message": f"Streaming error: {str(e)}"}
                yield f"data: {json.dumps(error_data)}\n\n"

        return Response(
            generate(),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
            },
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/dashboard/stats", methods=["GET"])
def get_dashboard_stats():
    """Get overall dashboard statistics"""
    try:
        stats = db_manager.get_dashboard_stats()

        # Add model-specific stats - Updated to query Message directly instead of Response
        session = db_manager.get_session()
        try:
            from database.models import Evaluation, Message

            # Model performance - Query evaluations joined with messages
            gpt_scores = (
                session.query(Evaluation.overall_score)
                .join(Message)
                .filter(Message.model_name == "gpt-5-nano", Evaluation.overall_score.isnot(None))
                .all()
            )
            gemini_scores = (
                session.query(Evaluation.overall_score)
                .join(Message)
                .filter(Message.model_name == "gemini-2.0-flash-lite", Evaluation.overall_score.isnot(None))
                .all()
            )

            gpt_avg = sum(s[0] for s in gpt_scores) / len(gpt_scores) if gpt_scores else 0
            gemini_avg = sum(s[0] for s in gemini_scores) / len(gemini_scores) if gemini_scores else 0

            # Tool success rate - No changes needed as ToolCall still references Message
            from database.models import ToolCall

            total_tools = session.query(ToolCall).count()
            successful_tools = session.query(ToolCall).filter(ToolCall.success == True).count()
            tool_success_rate = (successful_tools / total_tools * 100) if total_tools > 0 else 0

            stats.update(
                {
                    "gptScore": round(gpt_avg, 1),
                    "geminiScore": round(gemini_avg, 1),
                    "toolSuccessRate": round(tool_success_rate, 1),
                }
            )

        finally:
            session.close()

        return jsonify(stats)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/dashboard/conversations", methods=["GET"])
def get_conversations():
    """Get recent conversations"""
    try:
        limit = request.args.get("limit", 20, type=int)
        conversations = db_manager.get_conversations(limit)

        # Add additional info for each conversation - Updated to query Message directly
        session = db_manager.get_session()
        try:
            from database.models import Evaluation, Message

            for conv in conversations:
                # Get last message
                last_message = (
                    session.query(Message)
                    .filter(Message.conversation_id == conv["id"])
                    .order_by(Message.timestamp.desc())
                    .first()
                )

                if last_message:
                    conv["last_message"] = last_message.content[:100] + ("..." if len(last_message.content) > 100 else "")

                # Get message count
                conv["message_count"] = session.query(Message).filter(Message.conversation_id == conv["id"]).count()

                # Get average evaluation score - Updated to query evaluations directly through messages
                avg_score_query = (
                    session.query(Evaluation.overall_score)
                    .join(Message)
                    .filter(Message.conversation_id == conv["id"], Evaluation.overall_score.isnot(None))
                    .all()
                )

                if avg_score_query:
                    scores = [s[0] for s in avg_score_query if s[0] is not None]
                    conv["avg_score"] = round(sum(scores) / len(scores), 1) if scores else 0
                else:
                    conv["avg_score"] = 0

        finally:
            session.close()

        return jsonify(conversations)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/dashboard/evaluations", methods=["GET"])
def get_evaluations():
    """Get recent evaluations"""
    try:
        limit = request.args.get("limit", 20, type=int)

        session = db_manager.get_session()
        try:
            from database.models import Evaluation

            evaluations = session.query(Evaluation).order_by(Evaluation.timestamp.desc()).limit(limit).all()

            return jsonify([eval.to_dict() for eval in evaluations])

        finally:
            session.close()

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/evaluation/run", methods=["POST"])
def run_evaluation():
    """Run evaluation on unevaluated responses"""
    try:
        limit = request.json.get("limit", 50) if request.is_json else 50
        # evaluator.batch_evaluate_unevaluated(10)

        def run_background_evaluation():
            try:
                evaluator.batch_evaluate_unevaluated(limit)
            except Exception as e:
                print(f"Background evaluation error: {e}")

        # Run in background thread
        threading.Thread(target=run_background_evaluation, daemon=True).start()

        return jsonify({"status": "evaluation_started", "message": f"Evaluating up to {limit} responses"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/evaluation/status", methods=["GET"])
def get_evaluation_status():
    """Get evaluation status"""
    try:
        session = db_manager.get_session()
        try:
            from database.models import Evaluation, Message

            # Count assistant messages (these are what get evaluated)
            total_assistant_messages = session.query(Message).filter(Message.role == "assistant").count()

            # Count evaluated messages - Updated to count messages that have evaluations
            evaluated_messages = session.query(Message).filter(Message.role == "assistant").join(Evaluation).distinct().count()

            pending_evaluations = total_assistant_messages - evaluated_messages

            return jsonify(
                {
                    "total_assistant_messages": total_assistant_messages,
                    "evaluated_messages": evaluated_messages,
                    "pending_evaluations": pending_evaluations,
                    "evaluation_coverage": (
                        (evaluated_messages / total_assistant_messages * 100) if total_assistant_messages > 0 else 0
                    ),
                }
            )

        finally:
            session.close()

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/conversation/<int:conversation_id>", methods=["GET"])
def get_conversation_detail(conversation_id):
    """Get detailed conversation data"""
    try:
        # Use the simplified method from DatabaseManager
        conversation_data = db_manager.get_conversation_with_messages(conversation_id)

        if not conversation_data:
            return jsonify({"error": "Conversation not found"}), 404

        return jsonify(conversation_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/models", methods=["GET"])
def get_available_models():
    """Get list of available models"""
    return jsonify(
        {
            "models": [
                {
                    "id": "gpt-5-nano",
                    "name": "GPT-5 Nano",
                    "description": "Fast, Small model",
                    "provider": "OpenAI",
                },
                {
                    "id": "gemini-2.0-flash-lite",
                    "name": "Gemini-2.0 Flash Lite",
                    "description": "Fast and efficient for most tasks",
                    "provider": "Google",
                },
            ]
        }
    )


@app.route("/api/analytics/trends", methods=["GET"])
def get_analytics_trends():
    """Get analytics trends data"""
    try:
        days = request.args.get("days", 7, type=int)
        start_date = datetime.utcnow() - timedelta(days=days)

        session = db_manager.get_session()
        try:
            from sqlalchemy import Date, cast, func

            from database.models import Evaluation, Message

            # Get daily average scores - Updated to join Message directly instead of Response
            daily_scores = (
                session.query(
                    cast(Evaluation.timestamp, Date).label("date"),
                    func.avg(Evaluation.overall_score).label("avg_score"),
                    Message.model_name,
                )
                .join(Message)
                .filter(Evaluation.timestamp >= start_date, Evaluation.overall_score.isnot(None), Message.model_name.isnot(None))
                .group_by(cast(Evaluation.timestamp, Date), Message.model_name)
                .order_by("date")
                .all()
            )

            # Format data for charts
            dates = []
            gpt_scores = []
            gemini_scores = []

            # Group by date
            score_by_date = {}
            for score in daily_scores:
                date_str = score.date.strftime("%Y-%m-%d")
                if date_str not in score_by_date:
                    score_by_date[date_str] = {}
                score_by_date[date_str][score.model_name] = round(score.avg_score, 1)

            # Fill in the arrays
            for date_str in sorted(score_by_date.keys()):
                dates.append(date_str)
                gpt_scores.append(score_by_date[date_str].get("gpt-5-nano", 0))
                gemini_scores.append(score_by_date[date_str].get("gemini-2.0-flash-lite", 0))

            return jsonify({"dates": dates, "gpt_scores": gpt_scores, "gemini_scores": gemini_scores})

        finally:
            session.close()

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Create tables on startup
    try:
        from database.models import Base, db_manager

        # Tables are created in DatabaseManager.__init__
        print("Database initialized successfully")
    except Exception as e:
        print(f"Database initialization error: {e}")

    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)
