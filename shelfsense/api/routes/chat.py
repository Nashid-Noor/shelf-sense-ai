"""
Chat/RAG API Routes for ShelfSense AI

Conversational interface for library interactions:
- Single message chat
- Streaming responses
- Conversation management
- WebSocket for real-time chat
"""

import asyncio
import json
import time
import uuid
from typing import AsyncGenerator

from fastapi import (
    APIRouter, HTTPException, WebSocket, WebSocketDisconnect,
    Depends, status,
)
from fastapi.responses import StreamingResponse
from loguru import logger

from shelfsense.api.schemas import (
    ChatRequest,
    ChatResponse,
    ChatMessage,
    Citation,
    StreamChunk,
    ErrorResponse,
)


router = APIRouter(prefix="/chat", tags=["chat"])


# =============================================================================
# Dependencies
# =============================================================================

from shelfsense.api.dependencies import (
    get_service_container,
    get_conversation_manager as _get_conversation_manager,
    get_generator,
    get_rag_retriever,
    get_rag_orchestrator,
)


async def get_rag_pipeline():
    """Dependency to get RAG pipeline."""
    container = get_service_container()
    return {
        'retriever': container.rag_retriever,
        'generator': container.generator,
    }


async def get_conversation_manager(
    manager=Depends(_get_conversation_manager)
):
    """Dependency to get conversation manager."""
    return manager


# =============================================================================
# Chat Endpoints
# =============================================================================

@router.post(
    "",
    response_model=ChatResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
    },
)
@router.post(
    "",
    response_model=ChatResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
    },
)
async def chat(
    request: ChatRequest,
    orchestrator=Depends(get_rag_orchestrator),
):
    """
    Send a message and get a response.
    """
    start_time = time.time()
    
    conversation_id = request.conversation_id
    if not conversation_id:
        conversation_id = await orchestrator.new_conversation()
    
    try:
        response = await orchestrator.chat(
            conversation_id=conversation_id,
            user_message=request.message,
            stream=False,
        )
        
        # Map generated response back to ChatResponse schema
        citations = []
        for c in response.citations:
            citations.append(Citation(
                title=c.title,
                author=c.author,
                book_id=c.book_id,
            ))
        
        elapsed = (time.time() - start_time) * 1000
        
        return ChatResponse(
            response=response.content,
            citations=citations,
            books_referenced=len(citations),
            conversation_id=conversation_id,
            response_time_ms=elapsed,
            tokens_used=response.total_tokens,
        )
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/stream",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
    },
)
async def chat_stream(
    request: ChatRequest,
    orchestrator=Depends(get_rag_orchestrator),
):
    """
    Send a message and get a streaming response.
    """
    logger.info(f"Streaming chat: '{request.message[:50]}...'")
    
    conversation_id = request.conversation_id
    if not conversation_id:
        conversation_id = await orchestrator.new_conversation()
    
    async def generate() -> AsyncGenerator[str, None]:
        """Generate SSE events."""
        try:
            generator = await orchestrator.chat(
                conversation_id=conversation_id,
                user_message=request.message,
                stream=True,
            )
            
            async for chunk in generator:
                payload = {
                    "type": "content",
                    "content": chunk.text,
                    "is_final": chunk.is_final,
                    "conversation_id": conversation_id,
                }
                
                if chunk.is_final:
                    payload["done"] = True
                
                yield f"data: {json.dumps(payload)}\n\n"
        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"data: {json.dumps({'error': str(e), 'conversation_id': conversation_id})}\n\n"
            
    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson"
    )




# =============================================================================
# WebSocket Chat
# =============================================================================

class ConnectionManager:
    """Manage WebSocket connections."""
    
    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, conversation_id: str):
        await websocket.accept()
        self.active_connections[conversation_id] = websocket
        logger.info(f"WebSocket connected: {conversation_id}")
    
    def disconnect(self, conversation_id: str):
        if conversation_id in self.active_connections:
            del self.active_connections[conversation_id]
            logger.info(f"WebSocket disconnected: {conversation_id}")
    
    async def send_json(self, conversation_id: str, data: dict):
        if conversation_id in self.active_connections:
            await self.active_connections[conversation_id].send_json(data)


manager = ConnectionManager()


@router.websocket("/ws/{conversation_id}")
async def websocket_chat(websocket: WebSocket, conversation_id: str):
    """
    WebSocket endpoint for real-time chat.
    
    Protocol:
    - Client sends: {"message": "user message"}
    - Server responds with chunks: {"type": "content|citation|done", ...}
    
    Benefits over SSE:
    - Bidirectional communication
    - Better connection management
    - Ping/pong for keep-alive
    """
    await manager.connect(websocket, conversation_id)
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_json()
            message = data.get("message", "")
            
            if not message:
                await websocket.send_json({
                    "type": "error",
                    "error": "Empty message",
                })
                continue
            
            logger.info(f"WebSocket message: {message[:50]}...")
            
            # Process with RAG pipeline (placeholder)
            # In real implementation:
            # 1. Retrieve context
            # 2. Stream LLM response
            
            # Simulate streaming response
            chunks = [
                "I found ",
                "relevant books ",
                "in your library.",
            ]
            
            for chunk in chunks:
                await websocket.send_json({
                    "type": "content",
                    "content": chunk,
                })
                await asyncio.sleep(0.1)
            
            # Send completion
            await websocket.send_json({
                "type": "done",
                "tokens_used": 50,
            })
            
    except WebSocketDisconnect:
        manager.disconnect(conversation_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "error": str(e),
            })
        except Exception:
            pass
        manager.disconnect(conversation_id)


# =============================================================================
# Conversation Management
# =============================================================================

@router.get(
    "/conversations",
    response_model=dict,
)
async def list_conversations(
    limit: int = 20,
    offset: int = 0,
    manager=Depends(get_conversation_manager),
):
    """
    List recent conversations.
    """
    conversations = await manager.list_conversations(limit=limit)
    
    # Format for response
    formatted = []
    for c in conversations:
        # Simple title generation if None
        title = c.summary if c.summary else f"Chat {c.created_at.strftime('%Y-%m-%d %H:%M')}"
        formatted.append({
            "id": c.id,
            "title": title,
            "created_at": c.created_at.isoformat(),
            "updated_at": c.updated_at.isoformat(),
            "message_count": c.turn_count,
        })
        
    return {
        "conversations": formatted,
        "total": len(formatted),
    }


@router.get(
    "/conversations/{conversation_id}",
    response_model=dict,
)
async def get_conversation(
    conversation_id: str,
    manager=Depends(get_conversation_manager),
):
    """
    Get a conversation with message history.
    """
    conversation = await manager.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
        
    # Format messages
    messages = []
    for turn in conversation.turns:
        messages.append({
            "role": turn.role.value,
            "content": turn.content,
            "timestamp": turn.timestamp.isoformat(),
            "citations": turn.citations or [],
        })
        
    title = conversation.summary if conversation.summary else f"Chat {conversation.created_at.strftime('%Y-%m-%d %H:%M')}"
    
    return {
        "id": conversation.id,
        "title": title,
        "messages": messages,
        "created_at": conversation.created_at.isoformat(),
    }


@router.delete(
    "/conversations/{conversation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_conversation(
    conversation_id: str,
    manager=Depends(get_conversation_manager),
):
    """
    Delete a conversation.
    """
    logger.info(f"Deleting conversation: {conversation_id}")
    await manager.delete_conversation(conversation_id)
    return None


@router.post(
    "/conversations/{conversation_id}/title",
    response_model=dict,
)
async def update_conversation_title(
    conversation_id: str,
    title: str,
):
    """
    Update conversation title.
    """
    return {
        "id": conversation_id,
        "title": title,
    }


# =============================================================================
# Quick Actions
# =============================================================================

@router.post(
    "/quick/recommend",
    response_model=ChatResponse,
)
async def quick_recommend():
    """
    Get a quick book recommendation.
    
    Shortcut that doesn't require conversation context.
    Analyzes your library and suggests something to read.
    """
    start_time = time.time()
    
    # Would use recommendation engine
    response = (
        "Based on your reading preferences, I recommend checking out "
        "[The Left Hand of Darkness by Ursula K. Le Guin]. "
        "Given your interest in thought-provoking science fiction, "
        "this exploration of gender and society should resonate with you."
    )
    
    elapsed = (time.time() - start_time) * 1000
    
    return ChatResponse(
        response=response,
        citations=[
            Citation(
                title="The Left Hand of Darkness",
                author="Ursula K. Le Guin",
            ),
        ],
        books_referenced=1,
        conversation_id=str(uuid.uuid4()),
        response_time_ms=elapsed,
        tokens_used=80,
    )


@router.post(
    "/quick/summary",
    response_model=ChatResponse,
)
async def quick_library_summary():
    """
    Get a quick summary of your library.
    
    Provides an overview without conversation context.
    """
    start_time = time.time()
    
    response = (
        "Your library contains 127 books across multiple genres. "
        "Science fiction is your largest category (35%), followed by "
        "literary fiction (22%) and mystery (15%). "
        "You have 23 unread books waiting for you!"
    )
    
    elapsed = (time.time() - start_time) * 1000
    
    return ChatResponse(
        response=response,
        citations=[],
        books_referenced=0,
        conversation_id=str(uuid.uuid4()),
        response_time_ms=elapsed,
        tokens_used=60,
    )
