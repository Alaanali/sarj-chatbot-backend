import json
import time
from typing import Any, Dict, Optional

import openai

from database.models import db_manager


class LLMEvaluator:
    """LLM-based evaluation of chatbot responses"""

    def __init__(self, api_key: str, model: str = "gpt-5-nano"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    def evaluate_message(self, message_id: int) -> Optional[Dict[str, Any]]:
        """Evaluate a single assistant message using LLM"""

        # Get message data with context
        message_data = self._get_message_data(message_id)
        if not message_data:
            print(f"No message data found for message_id: {message_id}")
            return None

        # Check if already evaluated
        if self._is_already_evaluated(message_id):
            print(f"Message {message_id} already evaluated, skipping")
            return None

        evaluation_prompt = self._build_evaluation_prompt(message_data)

        try:
            start_time = time.time()

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert evaluator for weather chatbots. 
                        Evaluate responses on 5 dimensions (1-10 scale) and provide explanations.
                        Respond ONLY with valid JSON in the exact format specified.
                        Be objective and consider the context of weather assistance.""",
                    },
                    {"role": "user", "content": evaluation_prompt},
                ],
            )

            evaluation_time = round((time.time() - start_time) * 1000)

            # Parse evaluation result
            content = response.choices[0].message.content.strip()

            # Handle potential JSON parsing issues
            try:
                evaluation = json.loads(content)
            except json.JSONDecodeError as e:
                print(f"JSON parsing error for message {message_id}: {e}")
                print(f"Raw content: {content}")
                return None

            # Validate evaluation structure
            required_fields = [
                "helpfulness_score",
                "correctness_score",
                "politeness_score",
                "accuracy_score",
                "scope_adherence_score",
                "overall_score",
            ]

            if not all(field in evaluation for field in required_fields):
                print(f"Missing required fields in evaluation: {evaluation}")
                return None

            # Store in database
            eval_id = self._store_evaluation(message_id, evaluation, evaluation_time)
            evaluation["id"] = eval_id

            print(f"Successfully evaluated message {message_id} with score {evaluation['overall_score']}")
            return evaluation

        except Exception as e:
            print(f"Evaluation error for message {message_id}: {str(e)}")
            return None

    def _get_message_data(self, message_id: int) -> Optional[Dict[str, Any]]:
        """Get message data with context for evaluation"""
        session = db_manager.get_session()
        try:
            from database.models import Conversation, Message, ToolCall

            # Get the assistant message
            message = session.query(Message).filter(Message.id == message_id, Message.role == "assistant").first()
            if not message:
                return None

            # Get the conversation to find the preceding user message
            conversation = session.query(Conversation).filter(Conversation.id == message.conversation_id).first()
            if not conversation:
                return None

            # Get the user message that preceded this assistant message
            user_message = (
                session.query(Message)
                .filter(
                    Message.conversation_id == message.conversation_id,
                    Message.role == "user",
                    Message.timestamp < message.timestamp,
                )
                .order_by(Message.timestamp.desc())
                .first()
            )

            # Get tool calls for this message
            tool_calls = session.query(ToolCall).filter(ToolCall.message_id == message_id).all()

            return {
                "user_message": user_message.content if user_message else "No previous user message found",
                "response_content": message.content,
                "model_name": message.model_name,
                "response_time_ms": message.response_time_ms,
                "has_tool_calls": len(tool_calls) > 0,
                "error_occurred": message.error_occurred,
                "error_message": message.error_message,
                "tool_calls": [
                    {
                        "function_name": tc.function_name,
                        "arguments": tc.arguments,
                        "result": tc.result,
                        "success": tc.success,
                        "execution_time_ms": tc.execution_time_ms,
                    }
                    for tc in tool_calls
                ],
            }
        finally:
            session.close()

    def _is_already_evaluated(self, message_id: int) -> bool:
        """Check if message is already evaluated"""
        session = db_manager.get_session()
        try:
            from database.models import Evaluation

            return session.query(Evaluation).filter(Evaluation.message_id == message_id).first() is not None
        finally:
            session.close()

    def _build_evaluation_prompt(self, data: Dict[str, Any]) -> str:
        """Build evaluation prompt with message context"""

        # Format tool calls information
        tool_calls_text = ""
        if data["tool_calls"]:
            tool_calls_info = []
            for tc in data["tool_calls"]:
                status = "✅ Success" if tc["success"] else "❌ Failed"
                tool_calls_info.append(f"- {tc['function_name']}({tc['arguments']}) → {status} " f"({tc['execution_time_ms']}ms)")
            tool_calls_text = f"\nTool Calls Used:\n" + "\n".join(tool_calls_info)

        # Format error information
        error_text = ""
        if data["error_occurred"]:
            error_text = f"\nError Occurred: {data['error_message']}"

        return f"""
            Evaluate this weather chatbot interaction:

            USER QUERY: "{data['user_message']}"

            BOT RESPONSE: "{data['response_content']}"

            TECHNICAL DETAILS:
            - Model: {data['model_name']}
            - Response Time: {data['response_time_ms']}ms
            - Has Tool Calls: {data['has_tool_calls']}{tool_calls_text}{error_text}

            EVALUATION CRITERIA:
            Rate each dimension from 1-10 (where 10 is excellent, 1 is very poor):

            1. HELPFULNESS (1-10): How useful and informative is the response to the user?
            - Does it answer the user's question completely?
            - Is the information actionable and relevant?
            - Would a user find this response helpful?

            2. CORRECTNESS (1-10): Is the information factually accurate and appropriate?
            - Are there any factual errors or misconceptions?
            - Is the logic sound and reasoning correct?
            - Does the response make sense in context?

            3. POLITENESS (1-10): Is the tone professional, friendly, and respectful?
            - Is the language appropriate and courteous?
            - Does it maintain a helpful, non-judgmental tone?
            - Is it conversational yet professional?

            4. ACCURACY (1-10): For weather data, does it match expected information quality?
            - If weather data was retrieved, does it seem reasonable?
            - Are units, locations, and temporal references correct?
            - Is the data presentation clear and understandable?

            5. SCOPE_ADHERENCE (1-10): Does it stay focused on weather topics only?
            - Does it properly reject non-weather queries?
            - Does it redirect appropriately when off-topic?
            - Does it maintain focus on weather assistance?

            IMPORTANT: Respond with ONLY valid JSON in this exact format:
            {{
                "helpfulness_score": 8,
                "correctness_score": 9,
                "politeness_score": 10,
                "accuracy_score": 8,
                "scope_adherence_score": 9,
                "overall_score": 8.8,
                "helpfulness_explanation": "Brief explanation of helpfulness score",
                "correctness_explanation": "Brief explanation of correctness score", 
                "politeness_explanation": "Brief explanation of politeness score",
                "accuracy_explanation": "Brief explanation of accuracy score",
                "scope_adherence_explanation": "Brief explanation of scope adherence score",
                "overall_feedback": "Concise summary of overall performance and areas for improvement"
            }}

            The overall_score should be a weighted average emphasizing helpfulness and correctness most heavily.
        """

    def _store_evaluation(self, message_id: int, evaluation: Dict[str, Any], evaluation_time_ms: int) -> int:
        """Store evaluation results in database"""

        scores = {
            "helpfulness": evaluation["helpfulness_score"],
            "correctness": evaluation["correctness_score"],
            "politeness": evaluation["politeness_score"],
            "accuracy": evaluation["accuracy_score"],
            "scope_adherence": evaluation["scope_adherence_score"],
        }

        explanations = {
            "helpfulness": evaluation.get("helpfulness_explanation", ""),
            "correctness": evaluation.get("correctness_explanation", ""),
            "politeness": evaluation.get("politeness_explanation", ""),
            "accuracy": evaluation.get("accuracy_explanation", ""),
            "scope_adherence": evaluation.get("scope_adherence_explanation", ""),
        }

        return db_manager.create_evaluation(
            message_id=message_id,
            evaluator_model=self.model,
            scores=scores,
            explanations=explanations,
            overall_score=evaluation["overall_score"],
            overall_feedback=evaluation.get("overall_feedback", ""),
            evaluation_time_ms=evaluation_time_ms,
        )

    def batch_evaluate_unevaluated(self, limit: int = 50) -> Dict[str, Any]:
        """Evaluate all assistant messages that haven't been evaluated yet"""

        # Get unevaluated message IDs using the existing db_manager method
        message_ids = db_manager.get_unevaluated_assistant_messages(limit)

        if not message_ids:
            return {
                "status": "completed",
                "message": "No unevaluated assistant messages found",
                "evaluated_count": 0,
                "failed_count": 0,
            }

        print(f"Starting batch evaluation of {len(message_ids)} messages...")

        evaluated_count = 0
        failed_count = 0

        for i, message_id in enumerate(message_ids):
            try:
                print(f"Evaluating message {message_id} ({i+1}/{len(message_ids)})")

                evaluation = self.evaluate_message(message_id)
                if evaluation:
                    evaluated_count += 1
                else:
                    failed_count += 1

                # Rate limiting to avoid API limits
                if i < len(message_ids) - 1:  # Don't sleep after the last item
                    time.sleep(1)

            except Exception as e:
                print(f"Failed to evaluate message {message_id}: {e}")
                failed_count += 1
                continue

        result = {
            "status": "completed",
            "message": f"Batch evaluation completed: {evaluated_count} successful, {failed_count} failed",
            "evaluated_count": evaluated_count,
            "failed_count": failed_count,
            "total_processed": len(message_ids),
        }

        print(f"Batch evaluation completed: {evaluated_count} successful, {failed_count} failed")
        return result

    def get_evaluation_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get evaluation summary for the last N days"""
        from datetime import datetime, timedelta

        session = db_manager.get_session()
        try:
            from database.models import Evaluation, Message

            start_date = datetime.utcnow() - timedelta(days=days)

            evaluations = session.query(Evaluation).filter(Evaluation.timestamp >= start_date).all()

            if not evaluations:
                return {"message": "No evaluations found for the specified period"}

            # Calculate averages
            total_evals = len(evaluations)
            avg_helpfulness = sum(e.helpfulness_score for e in evaluations if e.helpfulness_score) / len(
                [e for e in evaluations if e.helpfulness_score]
            )
            avg_correctness = sum(e.correctness_score for e in evaluations if e.correctness_score) / len(
                [e for e in evaluations if e.correctness_score]
            )
            avg_politeness = sum(e.politeness_score for e in evaluations if e.politeness_score) / len(
                [e for e in evaluations if e.politeness_score]
            )
            avg_accuracy = sum(e.accuracy_score for e in evaluations if e.accuracy_score) / len(
                [e for e in evaluations if e.accuracy_score]
            )
            avg_scope = sum(e.scope_adherence_score for e in evaluations if e.scope_adherence_score) / len(
                [e for e in evaluations if e.scope_adherence_score]
            )
            avg_overall = sum(e.overall_score for e in evaluations if e.overall_score) / len(
                [e for e in evaluations if e.overall_score]
            )

            # Model breakdown - Updated to query Message instead of Response
            model_stats = {}
            for eval in evaluations:
                message = session.query(Message).filter(Message.id == eval.message_id).first()
                if message and message.model_name:
                    model = message.model_name
                    if model not in model_stats:
                        model_stats[model] = {"count": 0, "total_score": 0}
                    model_stats[model]["count"] += 1
                    model_stats[model]["total_score"] += eval.overall_score

            for model in model_stats:
                model_stats[model]["avg_score"] = model_stats[model]["total_score"] / model_stats[model]["count"]

            return {
                "period_days": days,
                "total_evaluations": total_evals,
                "average_scores": {
                    "helpfulness": round(avg_helpfulness, 2),
                    "correctness": round(avg_correctness, 2),
                    "politeness": round(avg_politeness, 2),
                    "accuracy": round(avg_accuracy, 2),
                    "scope_adherence": round(avg_scope, 2),
                    "overall": round(avg_overall, 2),
                },
                "model_performance": {
                    model: {"evaluations": stats["count"], "average_score": round(stats["avg_score"], 2)}
                    for model, stats in model_stats.items()
                },
            }
        finally:
            session.close()
