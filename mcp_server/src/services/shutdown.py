"""Graceful shutdown coordination for the episode queue.

On SIGTERM/SIGINT the server must not die with episodes still sitting in the
in-memory queue: they would be lost forever (the disk spool only fires on
processing *exceptions*). This coordinator wires the OS signals into a
three-step drain executed inside the running event loop:

1. ``QueueService.begin_stopping`` flips the stopping flag immediately, so any
   ``add_memory`` arriving during shutdown gets a backpressure rejection
   instead of being queued into a dying process.
2. The uvicorn server is asked to exit (bounded by
   ``timeout_graceful_shutdown``), while ``QueueService.drain`` spills every
   pending episode to the disk spool and gives the in-flight episode a grace
   window before cancelling its worker.
3. ``finalize`` awaits the drain result (idempotent) and stops the spool
   retryer, so the process can exit with an empty queue.

The signal handlers are installed *after* uvicorn captures its own, which
would otherwise swallow SIGTERM and wait indefinitely on long-lived
streamable-HTTP sessions.
"""

import asyncio
import logging
import signal
from typing import Any

logger = logging.getLogger(__name__)

HANDLED_SIGNALS = (signal.SIGTERM, signal.SIGINT)


class GracefulShutdownCoordinator:
    """Coordinate SIGTERM/SIGINT into a graceful drain of the episode queue."""

    def __init__(
        self,
        queue_service: Any,
        retryer: Any = None,
        server: Any = None,
        drain_timeout_seconds: float = 7.0,
    ):
        self.queue_service = queue_service
        self.retryer = retryer
        self.server = server
        self.drain_timeout_seconds = float(drain_timeout_seconds)
        self._drain_task: asyncio.Task | None = None
        self._signalled: signal.Signals | None = None

    # ---------------------------------------------------------------- signals

    def _on_signal(self, sig: signal.Signals) -> None:
        """Sync callback executed by the event loop on SIGTERM/SIGINT."""
        if self._signalled is not None:
            logger.warning('Received %s again during shutdown (first was %s); ignoring', sig.name, self._signalled.name)
            return
        self._signalled = sig
        logger.warning('Received %s: initiating graceful drain of episode queue', sig.name)
        # Instant backpressure: no new episodes are accepted from this point.
        self.queue_service.begin_stopping()
        asyncio.create_task(self._shutdown_sequence(), name='graceful-shutdown')

    async def _shutdown_sequence(self) -> None:
        """Ask the HTTP server to exit, then drain the queue concurrently."""
        try:
            if self.server is not None:
                self.server.should_exit = True
            self._drain_task = asyncio.create_task(
                self.queue_service.drain(wait_current_seconds=self.drain_timeout_seconds),
                name='episode-queue-drain',
            )
        except Exception as e:
            logger.error('Graceful shutdown sequence failed to start: %s', e)

    async def install_after(self, serve_task: asyncio.Task) -> None:
        """Install signal handlers once the uvicorn server has captured its own.

        uvicorn replaces any pre-existing handlers when ``serve()`` starts, so
        ours must be layered on top afterwards. Waits for ``server.started``
        (or the task finishing, e.g. on an early crash) before overriding.
        """
        deadline = asyncio.get_event_loop().time() + 10.0
        loop = asyncio.get_running_loop()
        while (
            self.server is not None
            and not getattr(self.server, 'started', False)
            and not serve_task.done()
            and loop.time() < deadline
        ):
            await asyncio.sleep(0.05)
        try:
            for sig in HANDLED_SIGNALS:
                loop.add_signal_handler(sig, self._on_signal, sig)
            logger.info('Graceful shutdown handlers installed for SIGTERM/SIGINT')
        except (NotImplementedError, RuntimeError, ValueError) as e:
            logger.warning('Could not install graceful shutdown signal handlers: %s', e)

    # ---------------------------------------------------------------- drain

    async def finalize(self) -> dict[str, Any] | None:
        """Ensure the drain completed and stop the retryer. Idempotent."""
        # Path without a prior signal (e.g. stdio EOF or a bare serve exit).
        self.queue_service.begin_stopping()
        if self._drain_task is None:
            self._drain_task = asyncio.create_task(
                self.queue_service.drain(wait_current_seconds=self.drain_timeout_seconds),
                name='episode-queue-drain',
            )
        try:
            result = await asyncio.shield(self._drain_task)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error('Episode queue drain failed: %s', e)
            result = None
        if self.retryer is not None:
            try:
                await self.retryer.stop()
            except Exception as e:
                logger.error('Failed to stop episode retryer during shutdown: %s', e)
        # Best-effort close of the journal-backed queue service: stop the journal
        # alarm and close the journal connection (durable rows survive anyway).
        try:
            await self.queue_service.close()
        except Exception as e:
            logger.error('Failed to close queue service during shutdown: %s', e)
        return result
