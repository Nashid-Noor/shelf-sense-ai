"""
Conversational Assistant Module

Retrieval-Augmented Generation for library conversations:
- Context-aware retrieval
- Source-grounded responses with citations
- Conversation memory management
- Multi-turn dialogue support
"""

from shelfsense.rag.retriever import (
    RAGRetriever,
    ConversationAwareRetriever,
    QueryExpander,
    ReRanker,
    RetrievedBook,
    RetrievalResult,
)
from shelfsense.rag.generator import (
    ResponseGenerator,
    GeneratedResponse,
    Citation,
    CitationExtractor,
    LLMProvider,
    AnthropicClient,
    OpenAIClient,
    GoogleClient,
    create_generator,
)
from shelfsense.rag.conversation import (
    ConversationManager,
    ConversationOrchestrator,
    ConversationStore,
    Conversation,
    ConversationTurn,
    ConversationContext,
    MessageRole,
)
from shelfsense.rag.prompts import PromptTemplates

__all__ = [
    # Retriever
    "RAGRetriever",
    "ConversationAwareRetriever",
    "QueryExpander",
    "ReRanker",
    "RetrievedBook",
    "RetrievalResult",
    # Generator
    "ResponseGenerator",
    "GeneratedResponse",
    "Citation",
    "CitationExtractor",
    "LLMProvider",
    "AnthropicClient",
    "OpenAIClient",
    "GoogleClient",
    "create_generator",
    # Conversation
    "ConversationManager",
    "ConversationOrchestrator",
    "ConversationStore",
    "Conversation",
    "ConversationTurn",
    "ConversationContext",
    "MessageRole",
    # Prompts
    "PromptTemplates",
]
