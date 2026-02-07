"""
RAG Prompts

Templates for system instructions, context injection, and response formatting.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PromptTemplates:
    """
    Collection of prompt templates for RAG.
    """
    
    # System prompt for the RAG assistant
    SYSTEM_PROMPT = """You are ShelfSense AI, an intelligent assistant for personal book libraries.

Your role:
- Help users explore and understand their book collection
- Answer questions about books in their library
- Provide recommendations based on their reading history
- Offer insights about reading patterns and preferences

Guidelines:
1. ONLY reference books from the provided context - never make up book information
2. If asked about a book not in the context, say "I don't see that book in your library"
3. When citing books, use the format [Title by Author]
4. Be conversational but informative
5. If the context doesn't contain enough information, acknowledge this
6. For recommendations, explain why based on their collection

Your knowledge is LIMITED to the books provided in the context below."""

    # Template for injecting retrieved context
    CONTEXT_TEMPLATE = """<library_context>
The following books are in the user's library:

{book_context}
</library_context>"""

    # Template for a single book in context
    BOOK_CONTEXT_TEMPLATE = """<book id="{book_id}">
Title: {title}
Author: {author}
{metadata}
</book>"""

    # Template for conversation history
    HISTORY_TEMPLATE = """<conversation_history>
{history}
</conversation_history>"""

    # Template for user message with context
    USER_MESSAGE_TEMPLATE = """{context}

{history}

User question: {question}

Remember: Only discuss books from the library context above. Cite sources as [Title by Author]."""

    # Response format instructions
    RESPONSE_FORMAT = """Format your response as follows:
1. Directly address the user's question
2. Reference specific books with citations [Title by Author]
3. If recommending, explain your reasoning
4. Keep responses concise but helpful"""

    # Few-shot examples for grounding
    FEW_SHOT_EXAMPLES = """Example interactions:

User: What science fiction books do I have?
Assistant: Based on your library, you have several science fiction titles including [Dune by Frank Herbert] and [Neuromancer by William Gibson]. Dune is a classic space opera about politics and ecology on a desert planet, while Neuromancer pioneered the cyberpunk genre.

User: Can you recommend something similar to 1984?
Assistant: While I don't see 1984 in your current library, if you enjoy dystopian fiction, you might appreciate [Brave New World by Aldous Huxley] which I do see in your collection. Both explore themes of societal control, though Huxley's vision focuses on pleasure and distraction rather than surveillance.

User: Tell me about the Harry Potter series
Assistant: I don't see any Harry Potter books in your library. Would you like me to discuss other fantasy series you do have, or help you identify similar books in your collection?"""

    # Recommendation prompt
    RECOMMENDATION_PROMPT = """Based on the user's library and preferences, recommend books they might enjoy.

Consider:
- Genres they seem to favor
- Authors they've collected multiple books from
- Themes and topics that appear frequently
- Any gaps or genres they might want to explore

{context}

User request: {question}

Provide 2-3 specific recommendations with explanations."""

    # Analysis prompt
    ANALYSIS_PROMPT = """Analyze the user's library to answer their question.

{context}

User question: {question}

Provide a thoughtful analysis based only on the books shown above."""

    # Summary prompt for collections
    COLLECTION_SUMMARY_PROMPT = """Summarize the user's book collection.

{context}

Provide:
1. Overview of collection size and diversity
2. Most represented genres/categories
3. Notable authors in the collection
4. Any interesting patterns or themes"""

    @classmethod
    def format_book_context(
        cls,
        books: list[dict],
        max_books: int = 10,
    ) -> str:
        """
        Format books into context string.
        
        Args:
            books: List of book dicts
            max_books: Maximum books to include
            
        Returns:
            Formatted context string
        """
        if not books:
            return "No books found in the relevant context."
        
        book_strings = []
        for book in books[:max_books]:
            # Build metadata string
            metadata_parts = []
            
            if book.get("genres"):
                genres = book["genres"][:3]  # Limit genres
                metadata_parts.append(f"Genres: {', '.join(genres)}")
            
            if book.get("publication_year"):
                metadata_parts.append(f"Published: {book['publication_year']}")
            
            if book.get("description"):
                # Truncate description
                desc = book["description"][:200]
                if len(book["description"]) > 200:
                    desc += "..."
                metadata_parts.append(f"Description: {desc}")
            
            metadata_str = "\n".join(metadata_parts) if metadata_parts else ""
            
            book_str = cls.BOOK_CONTEXT_TEMPLATE.format(
                book_id=book.get("id", "unknown"),
                title=book.get("title", "Unknown Title"),
                author=book.get("author", "Unknown Author"),
                metadata=metadata_str,
            )
            book_strings.append(book_str)
        
        return "\n\n".join(book_strings)
    
    @classmethod
    def format_conversation_history(
        cls,
        history: list[tuple[str, str]],
        max_turns: int = 5,
    ) -> str:
        """
        Format conversation history.
        
        Args:
            history: List of (role, message) tuples
            max_turns: Maximum turns to include
            
        Returns:
            Formatted history string
        """
        if not history:
            return ""
        
        # Take most recent turns
        recent = history[-max_turns:]
        
        formatted = []
        for role, message in recent:
            if role == "user":
                formatted.append(f"User: {message}")
            else:
                formatted.append(f"Assistant: {message}")
        
        return cls.HISTORY_TEMPLATE.format(
            history="\n".join(formatted)
        )
    
    @classmethod
    def build_prompt(
        cls,
        question: str,
        books: list[dict],
        history: Optional[list[tuple[str, str]]] = None,
        prompt_type: str = "default",
    ) -> tuple[str, str]:
        """
        Build complete prompt for RAG.
        
        Args:
            question: User's question
            books: Retrieved books
            history: Conversation history
            prompt_type: "default", "recommendation", "analysis", "summary"
            
        Returns:
            (system_prompt, user_prompt) tuple
        """
        # Format context
        book_context = cls.format_book_context(books)
        context = cls.CONTEXT_TEMPLATE.format(book_context=book_context)
        
        # Format history
        history_str = ""
        if history:
            history_str = cls.format_conversation_history(history)
        
        # Select template
        if prompt_type == "recommendation":
            user_prompt = cls.RECOMMENDATION_PROMPT.format(
                context=context,
                question=question,
            )
        elif prompt_type == "analysis":
            user_prompt = cls.ANALYSIS_PROMPT.format(
                context=context,
                question=question,
            )
        elif prompt_type == "summary":
            user_prompt = cls.COLLECTION_SUMMARY_PROMPT.format(
                context=context,
            )
        else:
            user_prompt = cls.USER_MESSAGE_TEMPLATE.format(
                context=context,
                history=history_str,
                question=question,
            )
        
        return cls.SYSTEM_PROMPT, user_prompt
    
    @classmethod
    def classify_intent(cls, question: str) -> str:
        """
        Classify user intent for prompt selection.
        
        Args:
            question: User's question
            
        Returns:
            Intent type: "recommendation", "analysis", "summary", "default"
        """
        question_lower = question.lower()
        
        # Recommendation keywords
        if any(word in question_lower for word in [
            "recommend", "suggest", "similar to", "like this",
            "what should i read", "what to read", "next book",
        ]):
            return "recommendation"
        
        # Analysis keywords
        if any(word in question_lower for word in [
            "analyze", "pattern", "trend", "statistics",
            "how many", "most common", "breakdown",
        ]):
            return "analysis"
        
        # Summary keywords
        if any(word in question_lower for word in [
            "summarize", "overview", "summary of",
            "tell me about my library", "my collection",
        ]):
            return "summary"
        
        return "default"


# Prompt for extracting citations from response
CITATION_EXTRACTION_PROMPT = """Extract all book citations from the following text.

Text: {text}

Return a JSON list of citations in the format:
[{{"title": "Book Title", "author": "Author Name"}}]

Only include books that are explicitly mentioned."""


# Prompt for fact-checking against context
FACT_CHECK_PROMPT = """Verify the following response against the provided context.

Context:
{context}

Response to verify:
{response}

Check if:
1. All mentioned books exist in the context
2. Book details (author, genre, etc.) are accurate
3. No information was hallucinated

Return a JSON object:
{{
    "is_valid": true/false,
    "issues": ["list of issues if any"],
    "confidence": 0.0-1.0
}}"""
