import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

from omegaconf import DictConfig
from openai.types.chat.chat_completion import ChatCompletion

from verl.protocol import DataProto
from verl.tools.base_tool import initialize_tools_from_config
from verl.utils.fs import copy_to_local
from verl.utils.tokenizer import hf_tokenizer
from verl.workers.rollout.chat_scheduler.chat_scheduler import ChatCompletionScheduler
from verl.workers.rollout.chat_scheduler.utils import QueueGroup


class CompletionCallback(ABC):
    def __init__(self, config: DictConfig, scheduler: "ChatCompletionScheduler"):
        self.config = config
        self.scheduler = scheduler

        # Initialize tools from config file
        self.max_turns = config.actor_rollout_ref.rollout.multi_turn.max_turns
        tool_config_path = config.actor_rollout_ref.rollout.multi_turn.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        self.tools = {tool.name: tool for tool in tool_list}
        self._tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]
        print(f"Initialized tools: {self.tools}", flush=True)

        local_path = copy_to_local(config.actor_rollout_ref.model.path)
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=True)

    @property
    def tool_schemas(self):
        """OpenAI JSON tool schemas."""
        return self._tool_schemas

    @property
    def extra_body(self) -> Dict[str, Any]:
        """Extra body pass to OpenAI API."""
        return None

    @abstractmethod
    async def __call__(self, messages: List[Dict[str, str]], completions: ChatCompletion, info: Dict[str, Any]):
        """Call back function to process completions.

        Args:
            messages: List of messages including raw prompt and assistant, tool response generated so far.
            completions: Chat completions from OpenAI compatible server.
            info: Any other auxiliary information pass across multi-turn.
        """
        raise NotImplementedError

    @abstractmethod
    def postprocess(self, batch: DataProto, batch_conversations: List[List[Dict[str, str]]], n: int) -> DataProto:
        """Post process batch data.

        Args:
            batch: Batch input messages from RLHFDataset.
            batch_conversations: List of messages including raw prompt, assistant response, tool response.
                Note that `len(batch_conversations) == len(batch) * n`, e.g n=2,
                batch_conversations=[messages_0_0, messages_0_1, messages_1_0, messages_1_1, ...]
            n: How many chat completion choices to generate for each input message.

        Returns:
            Batch data, should include ["prompts", "responses", "response_mask", "input_ids", "attention_mask", "position_ids"].
        """
        raise NotImplementedError


class Message:
    pass


@dataclass
class ActorMeta:
    actor_id: int
    queue_group: QueueGroup


class WorkFunc(Protocol):
    @abstractmethod
    async def __call__(self, meta: ActorMeta, message: Message) -> None:
        pass


@dataclass
class RolloutReq:
    completions: ChatCompletion
    info: Dict[str, Any]
    model_name: str
    messages: List[Dict[str, str]]
    chat_complete_request: Dict[str, Any]
    tools_schema: List[Dict[str, Any]]
    sampling_params: Dict[str, Any]
    extra_body: Dict[str, Any]


@dataclass
class RolloutResp:
    completions: ChatCompletion
    info: Dict[str, Any]
    messages: List[Dict[str, str]]
    exception: Optional[Exception] = None
    chat_complete_request: Dict[str, Any] = None
    model_name: str = None


@dataclass
class CallsReq:
    rollout_resp: RolloutResp
    actor_meta: ActorMeta


@dataclass
class ReduceResp:
    batch: DataProto
    batch_conversations: List[List[Dict[str, str]]]


class AsyncCallbackMixin(Protocol):
    def put(self, req: RolloutResp) -> bool:
        # this method will be called in coroutine
        # this method should act as a message queue
        ...

    def hit(self, req: RolloutResp) -> bool:
        # make sure this is a short function
        # this will be run in a coroutine
        ...


class CoroExternalCallsPlugin(AsyncCallbackMixin):
    def __init__(self, num_workers=50):
        self.plugin_queue = asyncio.Queue()
        self.num_workers = num_workers
        self._init_plugin_callers()

    def _init_plugin_callers(self):
        self.coros = [asyncio.create_task(self.run()) for _ in range(self.num_workers)]

    def put(self, req: RolloutResp) -> bool:
        self.plugin_queue.put(req)

    async def run(self):
        while True:
            req: CallsReq = await self.plugin_queue.get()
            result = self(req)
            id = req.actor_meta.actor_id
            req.actor_meta.queue_group.push(id, result)
