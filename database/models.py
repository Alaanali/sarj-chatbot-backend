from datetime import datetime

from sqlalchemy import (JSON, Boolean, Column, DateTime, Float, ForeignKey,
                        Integer, String, Text, create_engine)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

Base = declarative_base()


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True)
    session_id = Column(String(255), unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    user_ip = Column(String(45))
    user_agent = Column(Text)
    last_activity = Column(DateTime, default=datetime.utcnow)

    # Relationships
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan", order_by="Message.timestamp")

    @property
    def total_messages(self):
        """Total number of messages in this conversation"""
        return len(self.messages)

    @property
    def user_messages(self):
        """Filter for user messages only"""
        return [msg for msg in self.messages if msg.role == "user"]

    @property
    def assistant_messages(self):
        """Filter for assistant messages only"""
        return [msg for msg in self.messages if msg.role == "assistant"]

    @property
    def average_response_time(self):
        """Average response time for assistant messages"""
        response_times = [msg.response_time_ms for msg in self.assistant_messages if msg.response_time_ms]
        return sum(response_times) / len(response_times) if response_times else 0

    def to_dict(self):
        return {
            "id": self.id,
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "user_ip": self.user_ip,
            "user_agent": self.user_agent,
            "last_activity": self.last_activity.isoformat() if self.last_activity else None,
            "total_messages": self.total_messages,
            "average_response_time": self.average_response_time,
        }


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False)
    role = Column(String(20), nullable=False)  # 'user', 'assistant', 'system'
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

    # Assistant message specific fields (null for user messages)
    model_name = Column(String(100))
    response_time_ms = Column(Integer)
    tokens_used = Column(Integer)
    error_occurred = Column(Boolean, default=False)
    error_message = Column(Text)

    # Relationships
    conversation = relationship("Conversation", back_populates="messages")
    tool_calls = relationship("ToolCall", back_populates="message", cascade="all, delete-orphan")
    evaluations = relationship("Evaluation", back_populates="message", cascade="all, delete-orphan")

    @property
    def has_tool_calls(self):
        """Check if this message has associated tool calls"""
        return len(self.tool_calls) > 0

    @property
    def has_evaluations(self):
        """Check if this message has been evaluated"""
        return len(self.evaluations) > 0

    @property
    def tool_call_success_rate(self):
        """Success rate of tool calls for this message"""
        if not self.tool_calls:
            return None
        successful = sum(1 for tc in self.tool_calls if tc.success)
        return (successful / len(self.tool_calls)) * 100

    @property
    def average_evaluation_score(self):
        """Average overall evaluation score"""
        if not self.evaluations:
            return None
        scores = [eval.overall_score for eval in self.evaluations if eval.overall_score]
        return sum(scores) / len(scores) if scores else None

    def to_dict(self, include_relations=False):
        base_dict = {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "model_name": self.model_name,
            "response_time_ms": self.response_time_ms,
            "tokens_used": self.tokens_used,
            "error_occurred": self.error_occurred,
            "error_message": self.error_message,
            "has_tool_calls": self.has_tool_calls,
            "has_evaluations": self.has_evaluations,
            "tool_call_success_rate": self.tool_call_success_rate,
            "average_evaluation_score": self.average_evaluation_score,
        }

        if include_relations:
            base_dict.update(
                {
                    "tool_calls": [tc.to_dict() for tc in self.tool_calls],
                    "evaluations": [eval.to_dict() for eval in self.evaluations],
                }
            )

        return base_dict


class ToolCall(Base):
    __tablename__ = "tool_calls"

    id = Column(Integer, primary_key=True)
    message_id = Column(Integer, ForeignKey("messages.id"), nullable=False)
    function_name = Column(String(100), nullable=False)
    arguments = Column(JSON, nullable=False)
    result = Column(JSON)
    execution_time_ms = Column(Integer)
    success = Column(Boolean, default=True)
    error_message = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)

    # Relationships
    message = relationship("Message", back_populates="tool_calls")

    def to_dict(self):
        return {
            "id": self.id,
            "message_id": self.message_id,
            "function_name": self.function_name,
            "arguments": self.arguments,
            "result": self.result,
            "execution_time_ms": self.execution_time_ms,
            "success": self.success,
            "error_message": self.error_message,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }


class Evaluation(Base):
    __tablename__ = "evaluations"

    id = Column(Integer, primary_key=True)
    message_id = Column(Integer, ForeignKey("messages.id"), nullable=False)
    evaluator_model = Column(String(100), nullable=False)
    helpfulness_score = Column(Integer)
    correctness_score = Column(Integer)
    politeness_score = Column(Integer)
    accuracy_score = Column(Integer)
    scope_adherence_score = Column(Integer)
    overall_score = Column(Float)
    helpfulness_explanation = Column(Text)
    correctness_explanation = Column(Text)
    politeness_explanation = Column(Text)
    accuracy_explanation = Column(Text)
    scope_adherence_explanation = Column(Text)
    overall_feedback = Column(Text)
    evaluation_time_ms = Column(Integer)
    timestamp = Column(DateTime, default=datetime.utcnow)

    # Relationships
    message = relationship("Message", back_populates="evaluations")

    def to_dict(self):
        return {
            "id": self.id,
            "message_id": self.message_id,
            "evaluator_model": self.evaluator_model,
            "helpfulness_score": self.helpfulness_score,
            "correctness_score": self.correctness_score,
            "politeness_score": self.politeness_score,
            "accuracy_score": self.accuracy_score,
            "scope_adherence_score": self.scope_adherence_score,
            "overall_score": self.overall_score,
            "helpfulness_explanation": self.helpfulness_explanation,
            "correctness_explanation": self.correctness_explanation,
            "politeness_explanation": self.politeness_explanation,
            "accuracy_explanation": self.accuracy_explanation,
            "scope_adherence_explanation": self.scope_adherence_explanation,
            "overall_feedback": self.overall_feedback,
            "evaluation_time_ms": self.evaluation_time_ms,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }


# Database setup
class DatabaseManager:
    def __init__(self, db_url="sqlite:///chatbot_eval.db"):
        if "sqlite" in db_url and not db_url.endswith("?check_same_thread=False"):
            db_url += "?check_same_thread=False"

        self.engine = create_engine(
            db_url,
            echo=False,
            pool_timeout=20,
            pool_recycle=3600,
            pool_pre_ping=True,
            connect_args={"timeout": 30, "check_same_thread": False},
        )

        # Enable WAL mode
        with self.engine.begin() as conn:
            conn.exec_driver_sql("PRAGMA journal_mode=WAL;")
            conn.exec_driver_sql("PRAGMA synchronous=NORMAL;")

        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

    def get_session(self):
        return self.SessionLocal()

    def get_or_create_conversation(self, session_id, user_ip=None, user_agent=None):
        session = self.get_session()
        try:
            # Check if conversation exists
            conversation = session.query(Conversation).filter_by(session_id=session_id).first()

            if conversation:
                return conversation.id

            # Create new conversation if not found
            conversation = Conversation(session_id=session_id, user_ip=user_ip, user_agent=user_agent)
            session.add(conversation)
            session.commit()
            return conversation.id

        finally:
            session.close()

    def create_message(
        self, conversation_id, role, content, model_name=None, response_time_ms=None, error_occurred=False, error_message=None
    ):
        session = self.get_session()
        try:
            message = Message(
                conversation_id=conversation_id,
                role=role,
                content=content,
                model_name=model_name,
                response_time_ms=response_time_ms,
                error_occurred=error_occurred,
                error_message=error_message,
            )
            session.add(message)
            session.commit()

            # Update conversation last_activity
            conversation = session.query(Conversation).filter_by(id=conversation_id).first()
            if conversation:
                conversation.last_activity = datetime.utcnow()
                session.commit()

            return message.id
        finally:
            session.close()

    def create_tool_call(
        self, message_id, function_name, arguments, result, execution_time_ms=None, success=True, error_message=None
    ):
        session = self.get_session()
        try:
            tool_call = ToolCall(
                message_id=message_id,
                function_name=function_name,
                arguments=arguments,
                result=result,
                execution_time_ms=execution_time_ms,
                success=success,
                error_message=error_message,
            )
            session.add(tool_call)
            session.commit()
            return tool_call.id
        finally:
            session.close()

    def create_evaluation(
        self, message_id, evaluator_model, scores, explanations, overall_score, overall_feedback, evaluation_time_ms=None
    ):
        session = self.get_session()
        try:
            evaluation = Evaluation(
                message_id=message_id,
                evaluator_model=evaluator_model,
                helpfulness_score=scores.get("helpfulness"),
                correctness_score=scores.get("correctness"),
                politeness_score=scores.get("politeness"),
                accuracy_score=scores.get("accuracy"),
                scope_adherence_score=scores.get("scope_adherence"),
                overall_score=overall_score,
                helpfulness_explanation=explanations.get("helpfulness"),
                correctness_explanation=explanations.get("correctness"),
                politeness_explanation=explanations.get("politeness"),
                accuracy_explanation=explanations.get("accuracy"),
                scope_adherence_explanation=explanations.get("scope_adherence"),
                overall_feedback=overall_feedback,
                evaluation_time_ms=evaluation_time_ms,
            )
            session.add(evaluation)
            session.commit()
            return evaluation.id
        finally:
            session.close()

    def get_conversations(self, limit=50):
        session = self.get_session()
        try:
            conversations = session.query(Conversation).order_by(Conversation.last_activity.desc()).limit(limit).all()
            return [conv.to_dict() for conv in conversations]
        finally:
            session.close()

    def get_conversation_with_messages(self, conversation_id):
        session = self.get_session()
        try:
            conversation = session.query(Conversation).filter_by(id=conversation_id).first()
            if not conversation:
                return None

            result = conversation.to_dict()
            result["messages"] = [msg.to_dict(include_relations=True) for msg in conversation.messages]
            return result
        finally:
            session.close()

    def get_unevaluated_assistant_messages(self, limit=50):
        session = self.get_session()
        try:
            # Get assistant messages that don't have evaluations
            messages = (
                session.query(Message)
                .outerjoin(Evaluation)
                .filter(Message.role == "assistant", Evaluation.id.is_(None))
                .order_by(Message.timestamp.desc())
                .limit(limit)
                .all()
            )

            return [msg.id for msg in messages]
        finally:
            session.close()

    def get_dashboard_stats(self):
        session = self.get_session()
        try:
            # Basic counts
            total_conversations = session.query(Conversation).count()
            total_messages = session.query(Message).count()
            user_messages = session.query(Message).filter(Message.role == "user").count()
            assistant_messages = session.query(Message).filter(Message.role == "assistant").count()
            total_evaluations = session.query(Evaluation).count()

            # Average scores from evaluations
            evaluations = session.query(Evaluation).all()
            if evaluations:
                helpfulness_avg = sum(e.helpfulness_score for e in evaluations if e.helpfulness_score) / len(
                    [e for e in evaluations if e.helpfulness_score]
                )
                correctness_avg = sum(e.correctness_score for e in evaluations if e.correctness_score) / len(
                    [e for e in evaluations if e.correctness_score]
                )
                politeness_avg = sum(e.politeness_score for e in evaluations if e.politeness_score) / len(
                    [e for e in evaluations if e.politeness_score]
                )
                accuracy_avg = sum(e.accuracy_score for e in evaluations if e.accuracy_score) / len(
                    [e for e in evaluations if e.accuracy_score]
                )
                scope_avg = sum(e.scope_adherence_score for e in evaluations if e.scope_adherence_score) / len(
                    [e for e in evaluations if e.scope_adherence_score]
                )
                overall_avg = sum(e.overall_score for e in evaluations if e.overall_score) / len(
                    [e for e in evaluations if e.overall_score]
                )
            else:
                helpfulness_avg = correctness_avg = politeness_avg = accuracy_avg = scope_avg = overall_avg = 0

            # Average response time
            assistant_msgs = (
                session.query(Message).filter(Message.role == "assistant", Message.response_time_ms.isnot(None)).all()
            )
            avg_response_time = sum(msg.response_time_ms for msg in assistant_msgs) / len(assistant_msgs) if assistant_msgs else 0

            # Tool call success rate
            tool_calls = session.query(ToolCall).all()
            successful_tools = sum(1 for tc in tool_calls if tc.success)
            tool_success_rate = (successful_tools / len(tool_calls) * 100) if tool_calls else 0

            return {
                "totalConversations": total_conversations,
                "totalMessages": total_messages,
                "userMessages": user_messages,
                "assistantMessages": assistant_messages,
                "totalEvaluations": total_evaluations,
                "avgResponseTime": round(avg_response_time),
                "helpfulnessScore": round(helpfulness_avg, 1),
                "correctnessScore": round(correctness_avg, 1),
                "politenessScore": round(politeness_avg, 1),
                "accuracyScore": round(accuracy_avg, 1),
                "scopeScore": round(scope_avg, 1),
                "overallScore": round(overall_avg, 1),
                "toolSuccessRate": round(tool_success_rate, 1),
                "evaluationCoverage": round((total_evaluations / assistant_messages * 100) if assistant_messages > 0 else 0, 1),
            }
        finally:
            session.close()


# Global database manager instance
db_manager = DatabaseManager()
