"""
Ray-based executor for distributed evolution.

Enables scaling to clusters for massive parallelism ("Stress Tests").
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class RayExecutor:
    """
    Manages execution on a Ray cluster.
    """

    def __init__(self, num_workers: int, config_dict: dict, evaluation_file: str, env_vars: dict):
        self.num_workers = num_workers
        self.config_dict = config_dict
        self.evaluation_file = evaluation_file
        self.env_vars = env_vars
        self.futures = {}

        # Lazy import to avoid hard dependency
        try:
            import ray

            self.ray = ray
        except ImportError:
            raise ImportError("Ray is not installed. Please install with: pip install ray[default]")

        if not self.ray.is_initialized():
            logger.info("Initializing Ray...")
            # Use minimal resources to avoid blocking the head node if running locally
            self.ray.init(ignore_reinit_error=True)

        # Define the remote worker class
        @self.ray.remote
        class EvolutionWorker:
            def __init__(self, config_dict, evaluation_file, env_vars):
                import os

                os.environ.update(env_vars)

                # Setup global config in the worker process
                from openevolve.process_parallel import _worker_init

                _worker_init(config_dict, evaluation_file, env_vars)

            def run_iteration(self, iteration: int, snapshot: dict):
                from openevolve.process_parallel import _run_iteration_worker

                return _run_iteration_worker(iteration, snapshot)

        self.WorkerClass = EvolutionWorker
        self.actors = [
            self.WorkerClass.remote(config_dict, evaluation_file, env_vars)
            for _ in range(num_workers)
        ]
        self.actor_pool = list(self.actors)  # Simple round-robin or pool strategy

    def submit(self, iteration: int, snapshot: dict) -> Any:
        """Submit a task to Ray."""
        # Simple scheduling: pick an actor (Ray handles queuing on actors automatically)
        actor = self.actor_pool[iteration % len(self.actor_pool)]
        future = actor.run_iteration.remote(iteration, snapshot)
        self.futures[future] = iteration
        return RayFutureWrapper(future)

    def shutdown(self):
        """Shutdown Ray."""
        if self.ray.is_initialized():
            self.ray.shutdown()

    # Mimic Future interface for compatibility with ProcessParallelController logic?
    # ProcessParallelController uses concurrent.futures.Future.
    # Ray returns ObjectRef. We need a way to bridge them or modify Controller.

    # Actually, ProcessParallelController relies on `future.done()` and `future.result()`.
    # We should probably wrap Ray ObjectRef in a custom Future-like object.


class RayFutureWrapper:
    """Wraps a Ray ObjectRef to look like a concurrent.futures.Future."""

    def __init__(self, ref):
        self._ref = ref
        import ray

        self.ray = ray
        self._result = None
        self._exception = None
        self._done = False

    def cancel(self):
        self.ray.cancel(self._ref)
        return True

    def cancelled(self):
        return False

    def running(self):
        return not self._done

    def done(self):
        if self._done:
            return True
        # Check if ready
        ready, _ = self.ray.wait([self._ref], timeout=0)
        return bool(ready)

    def result(self, timeout=None):
        if self._done:
            if self._exception:
                raise self._exception
            return self._result

        try:
            self._result = self.ray.get(self._ref, timeout=timeout)
            self._done = True
            return self._result
        except Exception as e:
            self._exception = e
            self._done = True
            raise

    def exception(self, timeout=None):
        try:
            self.result(timeout)
            return None
        except Exception as e:
            return e

    def add_done_callback(self, fn):
        # Not fully supported in this simple wrapper without a background thread
        pass
