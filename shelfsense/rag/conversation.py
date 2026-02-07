"""
Conversation Manager for ShelfSense AI

Multi-turn dialogue management for RAG conversations:
- Session management with persistence
- Sliding window memory
- Context tracking (mentioned books)
- Conversation summarization

Design Decisions:
1. Sliding window: Keep last N turns to manage context length
2. Book tracking: Remember which books were discussed
3. Session persistence: Optional disk storage
4. Async-first: All I/O operations are async
"""

import asyncio
import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Any
from collections import deque
from enum import Enum

from loguru import logger


class MessageRole(str, Enum):
    """Role in conversation."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""
    
    id: str
    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Optional metadata
    query: Optional[str] = None  # Original user query
    retrieved_books: list[str] = field(default_factory=list)  # Book IDs used
    citations: list[str] = field(default_factory=list)  # Citations in response
    
    # Token counts
    token_count: int = 0
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "query": self.query,
            "retrieved_books": self.retrieved_books,
            "citations": self.citations,
            "token_count": self.token_count,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ConversationTurn":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            role=MessageRole(data["role"]),
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            query=data.get("query"),
            retrieved_books=data.get("retrieved_books", []),
            citations=data.get("citations", []),
            token_count=data.get("token_count", 0),
        )
    
    def to_tuple(self) -> tuple[str, str]:
        """Convert to (role, content) tuple for prompt building."""
        return (self.role.value, self.content)


@dataclass
class ConversationContext:
    """
    Contextual information for a conversation.
    
    Tracks mentioned books, topics, and user preferences
    discovered during the conversation.
    """
    
    # Books mentioned across conversation
    mentioned_books: list[str] = field(default_factory=list)
    
    # Topics discussed
    topics: list[str] = field(default_factory=list)
    
    # User preferences discovered
    preferences: dict = field(default_factory=dict)
    
    # Active filters (e.g., "only fiction")
    active_filters: dict = field(default_factory=dict)
    
    def add_mentioned_book(self, book_id: str):
        """Add a book to mentioned list."""
        if book_id not in self.mentioned_books:
            self.mentioned_books.append(book_id)
            # Keep reasonable size
            if len(self.mentioned_books) > 50:
                self.mentioned_books = self.mentioned_books[-50:]
    
    def add_topic(self, topic: str):
        """Add a discussion topic."""
        if topic not in self.topics:
            self.topics.append(topic)
            if len(self.topics) > 20:
                self.topics = self.topics[-20:]
    
    def to_dict(self) -> dict:
        """Serialize context."""
        return {
            "mentioned_books": self.mentioned_books,
            "topics": self.topics,
            "preferences": self.preferences,
            "active_filters": self.active_filters,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ConversationContext":
        """Deserialize context."""
        return cls(
            mentioned_books=data.get("mentioned_books", []),
            topics=data.get("topics", []),
            preferences=data.get("preferences", {}),
            active_filters=data.get("active_filters", {}),
        )


@dataclass
class Conversation:
    """
    A complete conversation session.
    
    Contains all turns and context for a single
    conversation with a user.
    """
    
    id: str
    user_id: Optional[str] = None
    
    # Conversation data
    turns: list[ConversationTurn] = field(default_factory=list)
    context: ConversationContext = field(default_factory=ConversationContext)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    # Summary for long conversations
    summary: Optional[str] = None
    
    # Status
    is_active: bool = True
    
    def add_turn(self, turn: ConversationTurn):
        """Add a turn to the conversation."""
        self.turns.append(turn)
        self.updated_at = datetime.utcnow()
        
        # Update context with mentioned books
        for book_id in turn.retrieved_books:
            self.context.add_mentioned_book(book_id)
    
    def get_recent_turns(self, n: int = 10) -> list[ConversationTurn]:
        """Get the N most recent turns."""
        return self.turns[-n:]
    
    def get_history_tuples(self, n: int = 5) -> list[tuple[str, str]]:
        """Get recent history as (role, content) tuples."""
        recent = self.get_recent_turns(n * 2)  # User + Assistant pairs
        return [turn.to_tuple() for turn in recent]
    
    def get_mentioned_books(self) -> list[str]:
        """Get all mentioned book IDs."""
        return self.context.mentioned_books
    
    @property
    def turn_count(self) -> int:
        """Number of turns in conversation."""
        return len(self.turns)
    
    @property
    def total_tokens(self) -> int:
        """Total tokens in conversation."""
        return sum(turn.token_count for turn in self.turns)
    
    def to_dict(self) -> dict:
        """Serialize conversation."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "turns": [turn.to_dict() for turn in self.turns],
            "context": self.context.to_dict(),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "summary": self.summary,
            "is_active": self.is_active,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Conversation":
        """Deserialize conversation."""
        return cls(
            id=data["id"],
            user_id=data.get("user_id"),
            turns=[ConversationTurn.from_dict(t) for t in data.get("turns", [])],
            context=ConversationContext.from_dict(data.get("context", {})),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            summary=data.get("summary"),
            is_active=data.get("is_active", True),
        )


class ConversationStore:
    """
    Persistent storage for conversations.
    
    Uses file-based JSON storage with optional async I/O.
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize conversation store.
        
        Args:
            storage_path: Directory for conversation files
        """
        self.storage_path = storage_path or Path("./conversations")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache
        self._cache: dict[str, Conversation] = {}
    
    def _get_filepath(self, conversation_id: str) -> Path:
        """Get file path for a conversation."""
        return self.storage_path / f"{conversation_id}.json"
    
    async def save(self, conversation: Conversation):
        """
        Save conversation to disk.
        
        Args:
            conversation: Conversation to save
        """
        # Update cache
        self._cache[conversation.id] = conversation
        
        # Write to disk asynchronously
        filepath = self._get_filepath(conversation.id)
        data = json.dumps(conversation.to_dict(), indent=2)
        
        await asyncio.to_thread(
            filepath.write_text,
            data,
        )
        
        logger.debug(f"Saved conversation {conversation.id}")
    
    async def load(self, conversation_id: str) -> Optional[Conversation]:
        """
        Load conversation from disk.
        
        Args:
            conversation_id: ID of conversation
            
        Returns:
            Conversation or None if not found
        """
        # Check cache first
        if conversation_id in self._cache:
            return self._cache[conversation_id]
        
        filepath = self._get_filepath(conversation_id)
        
        if not filepath.exists():
            return None
        
        try:
            data = await asyncio.to_thread(filepath.read_text)
            conversation = Conversation.from_dict(json.loads(data))
            
            # Update cache
            self._cache[conversation_id] = conversation
            
            return conversation
            
        except Exception as e:
            logger.error(f"Failed to load conversation {conversation_id}: {e}")
            return None
    
    async def delete(self, conversation_id: str):
        """
        Delete a conversation.
        
        Args:
            conversation_id: ID to delete
        """
        # Remove from cache
        if conversation_id in self._cache:
            del self._cache[conversation_id]
        
        # Delete file
        filepath = self._get_filepath(conversation_id)
        if filepath.exists():
            await asyncio.to_thread(filepath.unlink)
            logger.debug(f"Deleted conversation {conversation_id}")
    
    async def list_conversations(
        self,
        user_id: Optional[str] = None,
        limit: int = 50,
    ) -> list[Conversation]:
        """
        List recent conversations.
        
        Args:
            user_id: Filter by user
            limit: Maximum to return
            
        Returns:
            List of conversations
        """
        conversations = []
        
        for filepath in self.storage_path.glob("*.json"):
            try:
                data = json.loads(filepath.read_text())
                
                # Filter by user if specified
                if user_id and data.get("user_id") != user_id:
                    continue
                
                conversations.append(Conversation.from_dict(data))
                
            except Exception as e:
                logger.warning(f"Failed to load {filepath}: {e}")
        
        # Sort by updated_at descending
        conversations.sort(key=lambda c: c.updated_at, reverse=True)
        
        return conversations[:limit]


class ConversationManager:
    """
    Main conversation management interface.
    
    Handles:
    - Session lifecycle
    - Turn management
    - Context tracking
    - Memory windowing
    """
    
    def __init__(
        self,
        store: Optional[ConversationStore] = None,
        max_turns: int = 50,
        context_window_turns: int = 10,
        auto_summarize_threshold: int = 30,
    ):
        """
        Initialize conversation manager.
        
        Args:
            store: Persistence store (optional)
            max_turns: Maximum turns before truncation
            context_window_turns: Turns to include in context
            auto_summarize_threshold: Turns before auto-summarization
        """
        self.store = store
        self.max_turns = max_turns
        self.context_window_turns = context_window_turns
        self.auto_summarize_threshold = auto_summarize_threshold
        
        # Active conversations (in memory)
        self._active: dict[str, Conversation] = {}
    
    async def create_conversation(
        self,
        user_id: Optional[str] = None,
    ) -> Conversation:
        """
        Create a new conversation.
        
        Args:
            user_id: Optional user identifier
            
        Returns:
            New Conversation
        """
        conversation = Conversation(
            id=str(uuid.uuid4()),
            user_id=user_id,
        )
        
        self._active[conversation.id] = conversation
        
        # Persist if store configured
        if self.store:
            await self.store.save(conversation)
        
        logger.info(f"Created conversation {conversation.id}")
        return conversation
    
    async def get_conversation(
        self,
        conversation_id: str,
    ) -> Optional[Conversation]:
        """
        Get a conversation by ID.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            Conversation or None
        """
        # Check active first
        if conversation_id in self._active:
            return self._active[conversation_id]
        
        # Try loading from store
        if self.store:
            conversation = await self.store.load(conversation_id)
            if conversation:
                self._active[conversation_id] = conversation
                return conversation
        
        return None
    
    async def add_user_turn(
        self,
        conversation_id: str,
        content: str,
        query: Optional[str] = None,
    ) -> ConversationTurn:
        """
        Add a user turn to the conversation.
        
        Args:
            conversation_id: Conversation ID
            content: User message
            query: Optional original query
            
        Returns:
            Created turn
        """
        conversation = await self.get_conversation(conversation_id)
        if not conversation:
            raise ValueError(f"Conversation not found: {conversation_id}")
        
        turn = ConversationTurn(
            id=str(uuid.uuid4()),
            role=MessageRole.USER,
            content=content,
            query=query or content,
        )
        
        conversation.add_turn(turn)
        
        # Check for summarization
        await self._maybe_summarize(conversation)
        
        # Persist
        if self.store:
            await self.store.save(conversation)
        
        return turn
    
    async def add_assistant_turn(
        self,
        conversation_id: str,
        content: str,
        retrieved_books: Optional[list[str]] = None,
        citations: Optional[list[str]] = None,
        token_count: int = 0,
    ) -> ConversationTurn:
        """
        Add an assistant turn to the conversation.
        
        Args:
            conversation_id: Conversation ID
            content: Assistant response
            retrieved_books: Book IDs used in context
            citations: Citations in response
            token_count: Tokens in response
            
        Returns:
            Created turn
        """
        conversation = await self.get_conversation(conversation_id)
        if not conversation:
            raise ValueError(f"Conversation not found: {conversation_id}")
        
        turn = ConversationTurn(
            id=str(uuid.uuid4()),
            role=MessageRole.ASSISTANT,
            content=content,
            retrieved_books=retrieved_books or [],
            citations=citations or [],
            token_count=token_count,
        )
        
        conversation.add_turn(turn)
        
        # Persist
        if self.store:
            await self.store.save(conversation)
        
        return turn
    
    def get_context_for_generation(
        self,
        conversation: Conversation,
    ) -> dict:
        """
        Get context needed for response generation.
        
        Args:
            conversation: Current conversation
            
        Returns:
            Dict with history and context info
        """
        # Get recent turns for history
        history = conversation.get_history_tuples(self.context_window_turns)
        
        # Get mentioned books for retrieval boost
        mentioned_books = conversation.get_mentioned_books()
        
        return {
            "history": history,
            "mentioned_books": mentioned_books,
            "context": conversation.context,
            "summary": conversation.summary,
        }
    
    async def _maybe_summarize(self, conversation: Conversation):
        """
        Check if conversation needs summarization.
        
        Long conversations get summarized to manage context length.
        """
        if conversation.turn_count < self.auto_summarize_threshold:
            return
        
        if conversation.summary:
            # Already has summary, update periodically
            turns_since_summary = conversation.turn_count - self.auto_summarize_threshold
            if turns_since_summary < 10:
                return
        
        # Generate summary of older turns
        # Note: In production, this would call the LLM
        logger.info(
            f"Conversation {conversation.id} needs summarization "
            f"({conversation.turn_count} turns)"
        )
        
        # Simple extractive summary placeholder
        old_turns = conversation.turns[:-self.context_window_turns]
        topics = set()
        books = set()
        
        for turn in old_turns:
            if turn.citations:
                books.update(turn.citations)
        
        summary_parts = []
        if books:
            summary_parts.append(f"Books discussed: {', '.join(list(books)[:10])}")
        if conversation.context.topics:
            summary_parts.append(f"Topics: {', '.join(conversation.context.topics[:5])}")
        
        conversation.summary = " | ".join(summary_parts) if summary_parts else None
    
    async def end_conversation(self, conversation_id: str):
        """
        End a conversation.
        
        Args:
            conversation_id: Conversation to end
        """
        conversation = await self.get_conversation(conversation_id)
        if conversation:
            conversation.is_active = False
            
            if self.store:
                await self.store.save(conversation)
            
            # Remove from active
            if conversation_id in self._active:
                del self._active[conversation_id]
            
            logger.info(f"Ended conversation {conversation_id}")
    
    async def list_conversations(
        self,
        user_id: Optional[str] = None,
        limit: int = 50,
    ) -> list[Conversation]:
        """
        List recent conversations.
        
        Args:
            user_id: Filter by user
            limit: Maximum to return
            
        Returns:
            List of conversations
        """
        # If no store, return active ones sorted by time
        if not self.store:
            conversations = list(self._active.values())
            if user_id:
                conversations = [c for c in conversations if c.user_id == user_id]
            
            conversations.sort(key=lambda c: c.updated_at, reverse=True)
            return conversations[:limit]
            
        return await self.store.list_conversations(user_id, limit)

    async def delete_conversation(self, conversation_id: str):
        """
        Delete a conversation completely.
        """
        # Remove from active memory
        if conversation_id in self._active:
            del self._active[conversation_id]
            
        # Remove from store
        if self.store:
            await self.store.delete(conversation_id)
            
        logger.info(f"Deleted conversation {conversation_id}")

    async def clear_conversation(self, conversation_id: str):
        """
        Clear all turns from a conversation.
        
        Keeps the conversation but removes history.
        """
        conversation = await self.get_conversation(conversation_id)
        if conversation:
            conversation.turns = []
            conversation.context = ConversationContext()
            conversation.summary = None
            
            if self.store:
                await self.store.save(conversation)


class ConversationOrchestrator:
    """
    High-level orchestrator for RAG conversations.
    
    Combines retrieval, generation, and conversation management
    into a single interface.
    """
    
    def __init__(
        self,
        retriever,        # RAGRetriever
        generator,        # ResponseGenerator
        manager: ConversationManager,
    ):
        """
        Initialize orchestrator.
        
        Args:
            retriever: RAG retriever
            generator: Response generator
            manager: Conversation manager
        """
        self.retriever = retriever
        self.generator = generator
        self.manager = manager
    
    async def chat(
        self,
        conversation_id: str,
        user_message: str,
        stream: bool = False,
    ) -> Any:
        """
        Process a chat message.
        
        Args:
            conversation_id: Conversation ID
            user_message: User's message
            stream: Whether to stream response
            
        Returns:
            GeneratedResponse or async generator for streaming
        """
        # Get conversation
        conversation = await self.manager.get_conversation(conversation_id)
        if not conversation:
            raise ValueError(f"Conversation not found: {conversation_id}")
        
        # Add user turn
        await self.manager.add_user_turn(conversation_id, user_message)
        
        # Get context
        context = self.manager.get_context_for_generation(conversation)
        
        # Retrieve relevant books
        retrieval_result = await self.retriever.retrieve(
            query=user_message,
            conversation_context=context["mentioned_books"],
        )
        
        # Convert to dicts for generation
        retrieved_books = [book.to_dict() for book in retrieval_result.books]
        
        if stream:
            # Return streaming generator
            return self._stream_response(
                conversation_id,
                user_message,
                retrieved_books,
                context["history"],
            )
        else:
            # Generate complete response
            response = await self.generator.generate(
                query=user_message,
                retrieved_books=retrieved_books,
                conversation_history=context["history"],
            )
            
            # Add assistant turn
            await self.manager.add_assistant_turn(
                conversation_id,
                response.content,
                retrieved_books=[b["id"] for b in retrieved_books],
                citations=[str(c) for c in response.citations],
                token_count=response.total_tokens,
            )
            
            return response
    
    async def _stream_response(
        self,
        conversation_id: str,
        query: str,
        retrieved_books: list[dict],
        history: list[tuple[str, str]],
    ):
        """
        Stream a response and save when complete.
        """
        full_response = ""
        
        async for chunk in self.generator.generate_stream(
            query=query,
            retrieved_books=retrieved_books,
            conversation_history=history,
        ):
            full_response = chunk.accumulated_text
            yield chunk
            
            if chunk.is_final:
                # Save assistant turn
                await self.manager.add_assistant_turn(
                    conversation_id,
                    full_response,
                    retrieved_books=[b["id"] for b in retrieved_books],
                )
    
    async def new_conversation(
        self,
        user_id: Optional[str] = None,
    ) -> str:
        """
        Start a new conversation.
        
        Returns:
            New conversation ID
        """
        conversation = await self.manager.create_conversation(user_id)
        return conversation.id
