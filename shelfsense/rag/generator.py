"""
Response Generator

LLM integration for generating grounded responses with multi-provider support.
"""

import asyncio
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, AsyncGenerator, Any
from enum import Enum

from loguru import logger


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"

# ... (Previous code remains)




# ... (MockLLMClient remains)

def create_generator(
    provider: LLMProvider = LLMProvider.ANTHROPIC,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs,
) -> "ResponseGenerator":
    """
    Factory function to create a ResponseGenerator.
    """
    import os
    
    client = None
    
    # Try Anthropic
    if provider == LLMProvider.ANTHROPIC:
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if key:
            client = AnthropicClient(
                api_key=key,
                model=model or "claude-sonnet-4-20250514",
            )
        
    # Try OpenAI
    elif provider == LLMProvider.OPENAI:
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if key:
            client = OpenAIClient(
                api_key=key,
                model=model or "gpt-4-turbo-preview",
            )
            
    # Try Google
    elif provider == LLMProvider.GOOGLE:
        key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if key:
            client = GoogleClient(
                api_key=key,
                model=model or "gemini-1.5-flash",
            )
    
    # Fallback to Mock if no client created
    if client is None:
        logger.warning(f"No API keys found for provider {provider}. Using MockLLMClient.")
        client = MockLLMClient()
    
    # Import PromptTemplates
    from shelfsense.rag.prompts import PromptTemplates
    
    return ResponseGenerator(
        llm_client=client,
        prompt_templates=PromptTemplates,
        **kwargs,
    )


@dataclass
class Citation:
    """A citation extracted from a response."""
    
    title: str
    author: str
    book_id: Optional[str] = None
    position: int = 0  # Character position in response
    
    def __str__(self) -> str:
        return f"[{self.title} by {self.author}]"


@dataclass
class GeneratedResponse:
    """Complete response from the generator."""
    
    content: str
    citations: list[Citation] = field(default_factory=list)
    
    # Metadata
    model: str = ""
    provider: LLMProvider = LLMProvider.ANTHROPIC
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    # Validation
    is_grounded: bool = True
    ungrounded_claims: list[str] = field(default_factory=list)
    
    # Timing
    generation_time_ms: float = 0.0
    time_to_first_token_ms: float = 0.0


@dataclass
class StreamChunk:
    """A chunk from streaming response."""
    
    text: str
    is_final: bool = False
    accumulated_text: str = ""


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> GeneratedResponse:
        """Generate a complete response."""
        pass
    
    @abstractmethod
    async def generate_stream(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Generate response with streaming."""
        pass


class AnthropicClient(BaseLLMClient):
    """
    Anthropic Claude client for RAG generation.
    
    Supports Claude 3 family models with streaming.
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        default_max_tokens: int = 2048,
    ):
        """
        Initialize Anthropic client.
        
        Args:
            api_key: Anthropic API key
            model: Model identifier
            default_max_tokens: Default max output tokens
        """
        self.api_key = api_key
        self.model = model
        self.default_max_tokens = default_max_tokens
        self._client = None
    
    def _get_client(self):
        """Lazy initialization of Anthropic client."""
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.AsyncAnthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "anthropic package required. Install with: pip install anthropic"
                )
        return self._client
    
    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> GeneratedResponse:
        """
        Generate complete response using Claude.
        
        Args:
            system_prompt: System instructions
            user_prompt: User message with context
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            GeneratedResponse with content and metadata
        """
        import time
        start_time = time.time()
        
        client = self._get_client()
        
        try:
            response = await client.messages.create(
                model=self.model,
                max_tokens=max_tokens or self.default_max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ],
            )
            
            content = response.content[0].text
            
            elapsed_ms = (time.time() - start_time) * 1000
            
            return GeneratedResponse(
                content=content,
                model=self.model,
                provider=LLMProvider.ANTHROPIC,
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
                generation_time_ms=elapsed_ms,
            )
            
        except Exception as e:
            logger.error(f"Anthropic generation failed: {e}")
            raise
    
    async def generate_stream(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> AsyncGenerator[StreamChunk, None]:
        """
        Stream response using Claude.
        
        Args:
            system_prompt: System instructions
            user_prompt: User message
            max_tokens: Maximum tokens
            temperature: Sampling temperature
            
        Yields:
            StreamChunk objects
        """
        import time
        start_time = time.time()
        first_token_time = None
        
        client = self._get_client()
        accumulated = ""
        
        try:
            async with client.messages.stream(
                model=self.model,
                max_tokens=max_tokens or self.default_max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ],
            ) as stream:
                async for text in stream.text_stream:
                    if first_token_time is None:
                        first_token_time = time.time()
                    
                    accumulated += text
                    yield StreamChunk(
                        text=text,
                        is_final=False,
                        accumulated_text=accumulated,
                    )
            
            # Final chunk
            yield StreamChunk(
                text="",
                is_final=True,
                accumulated_text=accumulated,
            )
            
        except Exception as e:
            logger.error(f"Anthropic streaming failed: {e}")
            raise


class OpenAIClient(BaseLLMClient):
    """
    OpenAI GPT client for RAG generation.
    
    Supports GPT-4 family with streaming.
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4-turbo-preview",
        default_max_tokens: int = 2048,
    ):
        """
        Initialize OpenAI client.
        
        Args:
            api_key: OpenAI API key
            model: Model identifier
            default_max_tokens: Default max tokens
        """
        self.api_key = api_key
        self.model = model
        self.default_max_tokens = default_max_tokens
        self._client = None
    
    def _get_client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            try:
                import openai
                self._client = openai.AsyncOpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "openai package required. Install with: pip install openai"
                )
        return self._client
    
    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> GeneratedResponse:
        """Generate complete response using GPT."""
        import time
        start_time = time.time()
        
        client = self._get_client()
        
        try:
            response = await client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens or self.default_max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            
            content = response.choices[0].message.content
            
            elapsed_ms = (time.time() - start_time) * 1000
            
            return GeneratedResponse(
                content=content,
                model=self.model,
                provider=LLMProvider.OPENAI,
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                generation_time_ms=elapsed_ms,
            )
            
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise
    
    async def generate_stream(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream response using GPT."""
        client = self._get_client()
        accumulated = ""
        
        try:
            stream = await client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens or self.default_max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                stream=True,
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    text = chunk.choices[0].delta.content
                    accumulated += text
                    yield StreamChunk(
                        text=text,
                        is_final=False,
                        accumulated_text=accumulated,
                    )
            
            yield StreamChunk(
                text="",
                is_final=True,
                accumulated_text=accumulated,
            )
            
        except Exception as e:
            logger.error(f"OpenAI streaming failed: {e}")
            raise


class GoogleClient(BaseLLMClient):
    """
    Google Gemini client for RAG generation.
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "gemini-1.5-flash",
        default_max_tokens: int = 2048,
    ):
        self.api_key = api_key
        self.model = model
        self.default_max_tokens = default_max_tokens
        self._model = None
        
    def _get_model(self):
        """Lazy initialization of Google model."""
        if self._model is None:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._model = genai.GenerativeModel(self.model)
            except ImportError:
                raise ImportError(
                    "google-generativeai package required. Install with: pip install google-generativeai"
                )
        return self._model

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> GeneratedResponse:
        """Generate complete response using Gemini."""
        import time
        start_time = time.time()
        
        model = self._get_model()
        
        try:
            # Gemini handles system instructions differently, usually in config or combined prompt
            # For simplicity, we'll combine them here or use system_instruction if supported
            # gemini-1.5-flash supports system_instruction on init, but we are lazy loading model.
            # Best approach for per-request system prompt is combining.
            
            full_prompt = f"{system_prompt}\n\nUSER: {user_prompt}"
            
            response = await model.generate_content_async(
                full_prompt,
                generation_config={
                    "max_output_tokens": max_tokens or self.default_max_tokens,
                    "temperature": temperature,
                }
            )
            
            content = response.text
            
            elapsed_ms = (time.time() - start_time) * 1000
            
            return GeneratedResponse(
                content=content,
                model=self.model,
                provider=LLMProvider.GOOGLE,
                total_tokens=0,
                generation_time_ms=elapsed_ms,
            )
            
        except Exception as e:
            logger.error(f"Google generation failed: {e}")
            raise

    async def generate_stream(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream response using Gemini."""
        model = self._get_model()
        accumulated = ""
        
        try:
            full_prompt = f"{system_prompt}\n\nUSER: {user_prompt}"
            
            response = await model.generate_content_async(
                full_prompt,
                stream=True,
                generation_config={
                    "max_output_tokens": max_tokens or self.default_max_tokens,
                    "temperature": temperature,
                }
            )
            
            async for chunk in response:
                text = chunk.text
                accumulated += text
                yield StreamChunk(
                    text=text,
                    is_final=False,
                    accumulated_text=accumulated,
                )
            
            yield StreamChunk(text="", is_final=True, accumulated_text=accumulated)
            
        except Exception as e:
            logger.error(f"Google streaming failed: {e}")
            raise


class CitationExtractor:
    """
    Extract and validate citations from generated responses.
    
    Parses [Title by Author] format citations and matches
    them to books in the context.
    """
    
    # Pattern: [Title by Author] or [Title, Author]
    CITATION_PATTERN = re.compile(
        r'\[([^\[\]]+?)\s+by\s+([^\[\]]+?)\]|'  # [Title by Author]
        r'\[([^\[\]]+?),\s*([^\[\]]+?)\]'       # [Title, Author]
    )
    
    def __init__(self, context_books: Optional[list[dict]] = None):
        """
        Initialize extractor.
        
        Args:
            context_books: Books provided in context for validation
        """
        self.context_books = context_books or []
        self._build_lookup()
    
    def _build_lookup(self):
        """Build lookup index for validation."""
        self._title_lookup = {}
        self._author_lookup = {}
        
        for book in self.context_books:
            title_lower = book.get("title", "").lower()
            author_lower = book.get("author", "").lower()
            
            self._title_lookup[title_lower] = book
            
            if author_lower not in self._author_lookup:
                self._author_lookup[author_lower] = []
            self._author_lookup[author_lower].append(book)
    
    def extract(self, text: str) -> list[Citation]:
        """
        Extract citations from text.
        
        Args:
            text: Generated response text
            
        Returns:
            List of Citation objects
        """
        citations = []
        
        for match in self.CITATION_PATTERN.finditer(text):
            # Try first pattern [Title by Author]
            if match.group(1) and match.group(2):
                title = match.group(1).strip()
                author = match.group(2).strip()
            # Try second pattern [Title, Author]
            elif match.group(3) and match.group(4):
                title = match.group(3).strip()
                author = match.group(4).strip()
            else:
                continue
            
            citation = Citation(
                title=title,
                author=author,
                position=match.start(),
            )
            
            # Try to match to context book
            matched_book = self._match_to_context(title, author)
            if matched_book:
                citation.book_id = matched_book.get("id")
            
            citations.append(citation)
        
        return citations
    
    def _match_to_context(
        self,
        title: str,
        author: str,
    ) -> Optional[dict]:
        """
        Match citation to a book in context.
        
        Uses fuzzy matching for robustness.
        """
        title_lower = title.lower()
        author_lower = author.lower()
        
        # Exact title match
        if title_lower in self._title_lookup:
            return self._title_lookup[title_lower]
        
        # Fuzzy title match
        for ctx_title, book in self._title_lookup.items():
            # Check if one contains the other
            if title_lower in ctx_title or ctx_title in title_lower:
                # Verify author too
                if author_lower in book.get("author", "").lower():
                    return book
        
        # Author match with similar title
        if author_lower in self._author_lookup:
            author_books = self._author_lookup[author_lower]
            for book in author_books:
                if self._similar_titles(title_lower, book.get("title", "").lower()):
                    return book
        
        return None
    
    def _similar_titles(self, title1: str, title2: str) -> bool:
        """Check if titles are similar enough."""
        # Simple containment check
        if title1 in title2 or title2 in title1:
            return True
        
        # Word overlap
        words1 = set(title1.split())
        words2 = set(title2.split())
        overlap = len(words1 & words2) / max(len(words1), len(words2))
        
        return overlap > 0.5
    
    def validate_grounding(
        self,
        text: str,
        citations: list[Citation],
    ) -> tuple[bool, list[str]]:
        """
        Validate that response is properly grounded.
        
        Args:
            text: Response text
            citations: Extracted citations
            
        Returns:
            (is_grounded, list of ungrounded claims)
        """
        ungrounded = []
        
        for citation in citations:
            if citation.book_id is None:
                ungrounded.append(
                    f"Citation not in context: {citation}"
                )
        
        return len(ungrounded) == 0, ungrounded


class ResponseGenerator:
    """
    Main response generator orchestrating RAG generation.
    
    Features:
    - Provider-agnostic LLM interface
    - Automatic citation extraction
    - Grounding validation
    - Streaming support
    """
    
    def __init__(
        self,
        llm_client: BaseLLMClient,
        prompt_templates: Any,  # PromptTemplates class
        validate_grounding: bool = True,
        extract_citations: bool = True,
    ):
        """
        Initialize response generator.
        
        Args:
            llm_client: LLM client (Anthropic or OpenAI)
            prompt_templates: PromptTemplates for formatting
            validate_grounding: Whether to validate citations
            extract_citations: Whether to extract citations
        """
        self.llm_client = llm_client
        self.prompt_templates = prompt_templates
        self.validate_grounding = validate_grounding
        self.extract_citations = extract_citations
    
    async def generate(
        self,
        query: str,
        retrieved_books: list[dict],
        conversation_history: Optional[list[tuple[str, str]]] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> GeneratedResponse:
        """
        Generate a complete response.
        
        Args:
            query: User query
            retrieved_books: Books to include in context
            conversation_history: Previous turns
            max_tokens: Max output tokens
            temperature: Sampling temperature
            
        Returns:
            GeneratedResponse with citations
        """
        # Classify intent and build prompt
        intent = self.prompt_templates.classify_intent(query)
        system_prompt, user_prompt = self.prompt_templates.build_prompt(
            question=query,
            books=retrieved_books,
            history=conversation_history,
            prompt_type=intent,
        )
        
        # Generate response
        response = await self.llm_client.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        
        # Extract citations
        if self.extract_citations:
            extractor = CitationExtractor(retrieved_books)
            response.citations = extractor.extract(response.content)
            
            # Validate grounding
            if self.validate_grounding:
                is_grounded, ungrounded = extractor.validate_grounding(
                    response.content,
                    response.citations,
                )
                response.is_grounded = is_grounded
                response.ungrounded_claims = ungrounded
        
        return response
    
    async def generate_stream(
        self,
        query: str,
        retrieved_books: list[dict],
        conversation_history: Optional[list[tuple[str, str]]] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> AsyncGenerator[StreamChunk, None]:
        """
        Generate response with streaming.
        
        Args:
            query: User query
            retrieved_books: Context books
            conversation_history: Previous turns
            max_tokens: Max tokens
            temperature: Temperature
            
        Yields:
            StreamChunk objects
        """
        # Build prompt
        intent = self.prompt_templates.classify_intent(query)
        system_prompt, user_prompt = self.prompt_templates.build_prompt(
            question=query,
            books=retrieved_books,
            history=conversation_history,
            prompt_type=intent,
        )
        
        # Stream response
        async for chunk in self.llm_client.generate_stream(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        ):
            yield chunk
    
    async def generate_with_retry(
        self,
        query: str,
        retrieved_books: list[dict],
        max_retries: int = 2,
        **kwargs,
    ) -> GeneratedResponse:
        """
        Generate with automatic retry on failure.
        
        Args:
            query: User query
            retrieved_books: Context books
            max_retries: Maximum retry attempts
            **kwargs: Additional generation params
            
        Returns:
            GeneratedResponse
        """
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                return await self.generate(
                    query=query,
                    retrieved_books=retrieved_books,
                    **kwargs,
                )
            except Exception as e:
                last_error = e
                logger.warning(
                    f"Generation attempt {attempt + 1} failed: {e}"
                )
                
                if attempt < max_retries:
                    # Exponential backoff
                    await asyncio.sleep(2 ** attempt)
        
        raise last_error


class MockLLMClient(BaseLLMClient):
    """
    Mock client for development without API keys.
    """
    
    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> GeneratedResponse:
        """Generate a mock response."""
        content = (
            "I'm currently running in **offline mode** because no API keys were provided. "
            "I can see your books, but I can't generate new text about them.\n\n"
            "To enable full AI chat features, please add an `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` "
            "to your `.env` file."
        )
        
        return GeneratedResponse(
            content=content,
            model="mock-v1",
            provider=LLMProvider.ANTHROPIC, # Pretend to be anthropic structure
            generation_time_ms=10.0,
        )

    async def generate_stream(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream a mock response."""
        content = (
            "I'm currently running in **offline mode** because no API keys were provided. "
            "I can't generate new text about your books just yet.\n\n"
            "To enable full AI chat features, please configure your API keys."
        )
        
        tokens = content.split(" ")
        accumulated = ""
        
        for token in tokens:
            text = token + " "
            accumulated += text
            yield StreamChunk(
                text=text,
                is_final=False,
                accumulated_text=accumulated,
            )
            await asyncio.sleep(0.05)
            
        yield StreamChunk(text="", is_final=True, accumulated_text=accumulated)


def create_generator(
    provider: LLMProvider = LLMProvider.ANTHROPIC,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs,
) -> ResponseGenerator:
    """
    Factory function to create a ResponseGenerator.
    
    Args:
        provider: LLM provider
        api_key: API key (or from env)
        model: Model name (uses default if not specified)
        **kwargs: Additional ResponseGenerator params
        
    Returns:
        Configured ResponseGenerator
    """
    import os
    
    client = None
    
    # Try Anthropic
    if provider == LLMProvider.ANTHROPIC:
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if key:
            client = AnthropicClient(
                api_key=key,
                model=model or "claude-sonnet-4-20250514",
            )
        
    # Try OpenAI
    elif provider == LLMProvider.OPENAI:
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if key:
            client = OpenAIClient(
                api_key=key,
                model=model or "gpt-4-turbo-preview",
            )
    
    # Try Google
    elif provider == LLMProvider.GOOGLE:
        key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if key:
            client = GoogleClient(
                api_key=key,
                model=model or "gemini-1.5-flash",
            )

    # Fallback to Mock if no client created
    if client is None:
        logger.warning("No API keys found. Using MockLLMClient.")
        client = MockLLMClient()
    
    # Import PromptTemplates
    from shelfsense.rag.prompts import PromptTemplates
    
    return ResponseGenerator(
        llm_client=client,
        prompt_templates=PromptTemplates,
        **kwargs,
    )
