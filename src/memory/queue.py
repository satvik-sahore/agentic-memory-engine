import asyncio
import logging
import uuid
from typing import Optional, Dict, Any, Union, List
from pydantic import BaseModel, Field

from src.memory.models import MemoryScope

logger = logging.getLogger(__name__)


class IngestionJob(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    conversation: Union[str, List[Dict[str, Any]]]
    scope: MemoryScope = MemoryScope.USER
    session_id: Optional[str] = None
    workspace_id: Optional[str] = None
    status: str = "pending"
    operations_result: List[Any] = Field(default_factory=list)


class AsyncMemoryQueue:
    """Non-blocking background queue for asynchronous fact extraction and vector indexing."""

    def __init__(self):
        self._queue: Optional[asyncio.Queue] = None
        self._worker_task: Optional[asyncio.Task] = None
        self._is_running: bool = False
        self._jobs: Dict[str, IngestionJob] = {}

    def start(self):
        """Starts the background worker loop."""
        if self._is_running:
            return
        self._queue = asyncio.Queue()
        self._is_running = True
        self._worker_task = asyncio.create_task(self._worker_loop())
        logger.info("⚡ Async Memory Ingestion Queue worker started.")

    async def stop(self):
        """Gracefully drains remaining jobs and stops the worker."""
        if not self._is_running:
            return
        self._is_running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        logger.info("Async Memory Ingestion Queue stopped.")

    async def enqueue(
        self,
        user_id: str,
        conversation: Union[str, List[Dict[str, Any]]],
        scope: MemoryScope = MemoryScope.USER,
        session_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
    ) -> str:
        """Enqueues a new conversation ingestion job asynchronously without blocking."""
        if self._queue is None:
            self._queue = asyncio.Queue()

        job = IngestionJob(
            user_id=user_id,
            conversation=conversation,
            scope=scope,
            session_id=session_id,
            workspace_id=workspace_id,
            status="pending",
        )
        self._jobs[job.id] = job
        await self._queue.put(job)
        logger.debug(f"Enqueued async ingestion job [{job.id}] for user: {user_id}")
        return job.id

    def get_job_status(self, job_id: str) -> Optional[IngestionJob]:
        """Returns the current status of an async ingestion job."""
        return self._jobs.get(job_id)

    async def _worker_loop(self):
        """Continuous background worker processing ingestion jobs."""
        from src.memory.service import memory_service

        while self._is_running:
            try:
                if self._queue is None:
                    await asyncio.sleep(0.5)
                    continue

                job: IngestionJob = await self._queue.get()
                job.status = "processing"
                logger.info(f"⚡ Processing async memory job [{job.id}] for {job.user_id}...")

                # Run heavy extraction & reconciliation off the main request thread
                result = await asyncio.to_thread(
                    memory_service.process_conversation,
                    user_id=job.user_id,
                    conversation=job.conversation,
                    scope=job.scope,
                    session_id=job.session_id,
                    workspace_id=job.workspace_id,
                )

                job.status = "completed"
                job.operations_result = result.operations_performed
                self._queue.task_done()
                logger.info(f"✅ Async memory job [{job.id}] completed with {len(job.operations_result)} ops.")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in async memory worker loop: {e}", exc_info=True)
                await asyncio.sleep(1.0)


# Global singleton instance
async_memory_queue = AsyncMemoryQueue()
