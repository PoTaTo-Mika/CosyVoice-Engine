"""
Dynamic Batcher for CosyVoice LLM entry point.

Sits in front of the LLM server, collecting individual /v1/generate
requests and dispatching them as /v1/generate_batch calls.

Only the LLM layer needs dynamic batching — FM and Vocoder naturally
follow LLM's batch size since they process downstream output.

Batching strategy:
  - Scans the request queue every 0.2s
  - When 16 requests accumulate, dispatches immediately (no scan delay)
  - During low traffic, dispatches whatever is pending once the oldest
    request has waited >= 0.6s

Usage:
    python -m serve.tool_func.dynamic_batch \\
        --backend-url http://localhost:50000 --port 60000
"""

import argparse
import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


@dataclass
class PendingRequest:
    request_id: str
    payload: Dict[str, Any]
    future: asyncio.Future
    enqueue_time: float = field(default_factory=time.time)


class DynamicBatcher:
    """Collects individual /v1/generate requests and dispatches as batches."""

    def __init__(
        self,
        backend_url: str,
        max_batch_size: int = 16,
        scan_interval: float = 0.2,
        max_wait_time: float = 0.6,
        request_timeout: float = 60.0,
    ):
        self.backend_url = backend_url.rstrip('/')
        self.max_batch_size = max_batch_size
        self.scan_interval = scan_interval
        self.max_wait_time = max_wait_time
        self.request_timeout = request_timeout

        self.queue: asyncio.Queue[PendingRequest] = asyncio.Queue()
        self._scanner_task: Optional[asyncio.Task] = None
        self._client: Optional[httpx.AsyncClient] = None

        # Stats
        self.total_requests = 0
        self.total_batches = 0
        self.total_errors = 0

    async def start(self):
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(self.request_timeout))
        self._scanner_task = asyncio.create_task(self._scanner_loop())
        logger.info(
            f'DynamicBatcher started: backend={self.backend_url}, '
            f'max_batch={self.max_batch_size}, '
            f'scan={self.scan_interval}s, max_wait={self.max_wait_time}s'
        )

    async def stop(self):
        if self._scanner_task:
            self._scanner_task.cancel()
            try:
                await self._scanner_task
            except asyncio.CancelledError:
                pass
        if self._client:
            await self._client.aclose()

    async def submit(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Submit a single request, wait for batched result."""
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        req = PendingRequest(
            request_id=uuid.uuid4().hex[:8],
            payload=payload,
            future=future,
        )
        self.total_requests += 1
        self.queue.put_nowait(req)

        # Immediate dispatch trigger when queue is full
        if self.queue.qsize() >= self.max_batch_size:
            asyncio.create_task(self._try_dispatch())

        try:
            return await asyncio.wait_for(future, timeout=self.request_timeout)
        except asyncio.TimeoutError:
            self.total_errors += 1
            raise RuntimeError(
                f'Request {req.request_id} timed out after {self.request_timeout}s'
            )

    # ------------------------------------------------------------------
    # Scanner loop
    # ------------------------------------------------------------------

    async def _scanner_loop(self):
        while True:
            try:
                await asyncio.sleep(self.scan_interval)
                await self._try_dispatch()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.error('Scanner error', exc_info=True)

    async def _try_dispatch(self):
        """Drain queue, dispatch if conditions are met, otherwise put back."""
        pending: list[PendingRequest] = []
        while not self.queue.empty():
            try:
                pending.append(self.queue.get_nowait())
            except asyncio.QueueEmpty:
                break

        if not pending:
            return

        now = time.time()
        oldest_wait = now - pending[0].enqueue_time
        should_dispatch = (
            len(pending) >= self.max_batch_size
            or oldest_wait >= self.max_wait_time
        )

        if not should_dispatch:
            for req in pending:
                self.queue.put_nowait(req)
            return

        # Dispatch in chunks of max_batch_size
        for i in range(0, len(pending), self.max_batch_size):
            batch = pending[i:i + self.max_batch_size]
            asyncio.create_task(self._dispatch_batch(batch))

    async def _dispatch_batch(self, batch: list[PendingRequest]):
        """Send a batch to the backend and resolve individual futures."""
        self.total_batches += 1
        items = [req.payload for req in batch]
        batch_id = self.total_batches
        wait = time.time() - batch[0].enqueue_time

        logger.info(
            f'[Batch #{batch_id}] dispatching {len(batch)} requests '
            f'(oldest waited {wait:.3f}s)'
        )

        try:
            resp = await self._client.post(
                f'{self.backend_url}/v1/generate_batch',
                json={'items': items},
            )
            resp.raise_for_status()
            result = resp.json()

            results = result.get('results', [])
            if len(results) != len(batch):
                raise RuntimeError(
                    f'Backend returned {len(results)} results for {len(batch)} requests'
                )

            for i, req in enumerate(batch):
                if not req.future.done():
                    req.future.set_result(results[i])

        except Exception as e:
            self.total_errors += 1
            logger.error(f'[Batch #{batch_id}] failed: {e}')
            for req in batch:
                if not req.future.done():
                    req.future.set_exception(e)


# ------------------------------------------------------------------
# FastAPI app
# ------------------------------------------------------------------

def create_app(batcher: DynamicBatcher) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        await batcher.start()
        yield
        await batcher.stop()

    app = FastAPI(title='CosyVoice Dynamic Batcher', lifespan=lifespan)

    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=['*'], allow_methods=['*'], allow_headers=['*'],
    )

    @app.post('/v1/generate')
    async def generate(request: Request):
        payload = await request.json()
        result = await batcher.submit(payload)
        return JSONResponse(content=result)

    @app.post('/v1/generate_batch')
    async def generate_batch(request: Request):
        """Pass-through: pre-batched requests go directly to backend."""
        payload = await request.json()
        resp = await batcher._client.post(
            f'{batcher.backend_url}/v1/generate_batch',
            json=payload,
        )
        return JSONResponse(content=resp.json())

    @app.get('/health')
    async def health():
        queue_size = batcher.queue.qsize()
        backend_ok = False
        try:
            resp = await batcher._client.get(
                f'{batcher.backend_url}/health', timeout=3.0
            )
            backend_ok = resp.status_code == 200
        except Exception:
            pass
        return {
            'status': 'ok' if backend_ok else 'degraded',
            'backend_url': batcher.backend_url,
            'backend_healthy': backend_ok,
            'max_batch_size': batcher.max_batch_size,
            'scan_interval': batcher.scan_interval,
            'max_wait_time': batcher.max_wait_time,
            'queue_size': queue_size,
            'total_requests': batcher.total_requests,
            'total_batches': batcher.total_batches,
            'total_errors': batcher.total_errors,
        }

    return app


def main():
    parser = argparse.ArgumentParser(description='CosyVoice Dynamic Batcher')
    parser.add_argument('--backend-url', type=str, required=True,
                        help='LLM backend URL (e.g. http://localhost:50000)')
    parser.add_argument('--port', type=int, required=True,
                        help='Port to listen on')
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--max-batch-size', type=int, default=16)
    parser.add_argument('--scan-interval', type=float, default=0.2,
                        help='Queue scan interval in seconds')
    parser.add_argument('--max-wait-time', type=float, default=0.6,
                        help='Max time a request waits before partial batch dispatch')
    parser.add_argument('--request-timeout', type=float, default=60.0,
                        help='Timeout for individual request (seconds)')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s [%(name)s] %(message)s',
    )

    batcher = DynamicBatcher(
        backend_url=args.backend_url,
        max_batch_size=args.max_batch_size,
        scan_interval=args.scan_interval,
        max_wait_time=args.max_wait_time,
        request_timeout=args.request_timeout,
    )
    app = create_app(batcher)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
