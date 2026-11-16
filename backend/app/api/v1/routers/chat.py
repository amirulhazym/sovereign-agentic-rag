"""REST & SSE streaming routers for Agentic RAG chat and session history."""

from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from app.core.dependencies import get_chat_service, get_sqlite_repo
from app.repositories.sqlite_repo import SQLiteRepository
from app.schemas.chat import ChatStreamRequest
from app.schemas.workspace import ChatMessageRead, ChatSessionRead
from app.services.chat_service import ChatService

router = APIRouter(prefix="/api/v1", tags=["chat"])


@router.post(
    "/chat/stream",
    summary="Stream bounded LangGraph node transitions, answer tokens, and citations via SSE",
)
async def stream_chat(
    payload: ChatStreamRequest,
    request: Request,
    chat_service: Annotated[ChatService, Depends(get_chat_service)],
) -> StreamingResponse:
    """Validate workspace before opening SSE stream; emit typed frames & monitor disconnect."""
    collection, session_id, chat_history, mode = await chat_service.prepare_stream(payload)
    return StreamingResponse(
        chat_service.stream_events(
            request=payload,
            collection=collection,
            session_id=session_id,
            chat_history=chat_history,
            mode=mode,
            is_disconnected=request.is_disconnected,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.get(
    "/collections/{collection_id}/sessions",
    response_model=list[ChatSessionRead],
    summary="List chat sessions inside a collection",
)
async def list_collection_sessions(
    collection_id: str,
    sqlite_repo: Annotated[SQLiteRepository, Depends(get_sqlite_repo)],
) -> list[ChatSessionRead]:
    """Return all conversation threads belonging to `collection_id`."""
    return await sqlite_repo.list_collection_sessions(collection_id)


@router.get(
    "/sessions/{session_id}/messages",
    response_model=list[ChatMessageRead],
    summary="List persisted messages for a chat session",
)
async def list_session_messages(
    session_id: str,
    sqlite_repo: Annotated[SQLiteRepository, Depends(get_sqlite_repo)],
) -> list[ChatMessageRead]:
    """Return chronological chat turns including citations and cache telemetry."""
    return await sqlite_repo.list_session_messages(session_id)
