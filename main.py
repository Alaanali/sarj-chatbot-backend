import json

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, request
from flask_cors import CORS

from models import get_model

load_dotenv()

app = Flask(__name__)
CORS(app)


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

        if not message:
            return jsonify({"error": "Message is required"}), 400

        if model not in ["gpt-5-nano", "gemini-2.0-flash-lite"]:
            return jsonify({"error": f"Unsupported model: {model}"}), 400

        handler = get_model(model)
        if not handler:
            return jsonify({"error": f"Unsupported model: {model}"}), 400

        # Create streaming response
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


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)
