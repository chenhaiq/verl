import asyncio
import functools
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from openai.types.chat.chat_completion import ChatCompletion

from verl.protocol import DataProto
from verl.workers.rollout.chat_scheduler.utils import ActorMeta


@dataclass
class RolloutReq:
    completions: ChatCompletion
    model_name: str
    messages: List[Dict[str, str]]
    tools_schema: List[Dict[str, Any]]
    sampling_params: Dict[str, Any]
    extra_body: Dict[str, Any]


@dataclass
class RolloutResp:
    completions: ChatCompletion
    messages: List[Dict[str, str]]
    exception: Optional[Exception] = None
    sampling_params: Dict[str, Any] = None
    model_name: str = None


@dataclass
class CallsReq:
    rollout_resp: RolloutResp
    actor_meta: ActorMeta


@dataclass
class ReduceResp:
    batch: DataProto
    batch_conversations: List[List[Dict[str, str]]]


@runtime_checkable
class AsyncCallbackMixin(Protocol):
    def put(self, req: RolloutResp) -> bool:
        # this method will be called in coroutine
        # this method should act as a message queue
        ...

    def hit(self, req: RolloutResp) -> bool:
        # make sure this is a short function
        # this will be run in a coroutine
        ...

    async def shutdown(self): ...


class CoroExternalCallsPlugin(AsyncCallbackMixin):
    def __init__(self, num_workers=3):
        self.plugin_queue = asyncio.Queue()
        self.num_workers = num_workers
        self._init_plugin_callers()
        self.shut_down_flag = False
        self.shutdown_evt = asyncio.Event()

    def _init_plugin_callers(self):
        print("init plugin callers for CoroExternalCallsPlugin with worker: ", self.num_workers)
        self.coros = [asyncio.create_task(self.run()) for _ in range(self.num_workers)]

    def put(self, req: RolloutResp) -> bool:
        self.plugin_queue.put(req)

    async def shutdown(self):
        self.shut_down_flag = True

        def set_evt(task: asyncio.Task, evt: asyncio.Event):
            evt.set()

        evts = []
        for coro in self.coros:
            if not coro.done() or not coro.cancelled():
                evt = asyncio.Event()
                evts.append(evt.wait())
                coro.add_done_callback(functools.partial(set_evt, evt=evt))
                coro.cancel()
        print("waiting for CoroExternalCallsPlugin to shutdown, with length: ", len(evts))
        await asyncio.gather(*evts)
        print("shutdown coros for CoroExternalCallsPlugin")

    async def run(self):
        while not self.shut_down_flag:
            req: CallsReq = await self.plugin_queue.get()
            result = self(req)
            id = req.actor_meta.actor_id
            req.actor_meta.queue_group.push(id, result)
