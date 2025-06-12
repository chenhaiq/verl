# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import functools
import heapq
import importlib
import logging
import threading
from typing import Any, Dict, List, Optional
from uuid import uuid4

import aiohttp
from cachetools import LRUCache
from omegaconf import DictConfig
from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion

from verl.protocol import DataProto
from verl.workers.rollout.chat_scheduler.apis import AsyncCallbackMixin, CallsReq, RolloutReq, RolloutResp
from verl.workers.rollout.chat_scheduler.callback import ToolCompletionCallback
from verl.workers.rollout.chat_scheduler.utils import ActorMeta, QueueGroup, WorkStealingActor

logger = logging.getLogger(__file__)


class ChatCompletionScheduler:
    def __init__(
        self,
        config: DictConfig,
        server_addresses: List[str],
        max_cache_size: int = 10000,
    ):
        """
        Args:
            config: DictConfig.
            server_addresses: List[str], OpenAI compatible server addresses.
            max_cache_size: int, max cache size of request_id to address mapping.
        """
        self.config = config.actor_rollout_ref.rollout
        model_path = config.actor_rollout_ref.model.path
        self.model_name = "/".join(model_path.split("/")[-2:])

        # Least requests load balancing
        self.weighted_addresses = [[0, address] for address in server_addresses]
        heapq.heapify(self.weighted_addresses)

        # LRU cache to map request_id to address
        self.request_id_to_address = LRUCache(maxsize=max_cache_size)

        self.background_tasks = set()
        if self.config.multi_turn.completion_callback is None:
            self.completion_callback = ToolCompletionCallback(config, self)
            logger.warning("completion_callback is None, use ToolCompletionCallback")
        else:
            module_path, class_name = self.config.multi_turn.completion_callback.rsplit(".", 1)
            module = importlib.import_module(module_path)
            self.completion_callback = getattr(module, class_name)(config, self)

    def submit_chat_completions(self, *, messages: List[Dict[str, str]], request_id: str, info: Dict[str, Any], address: Optional[str] = None):
        """Submit chat completion request without wait, completion_callback will be called when the request is done.

        Args:
            messages: List of messages.
            request_id: Request id.
            info: Any other auxiliary information pass across multi-turn.
        """
        info["__depth__"] += 1
        task = asyncio.create_task(self._submit_chat_completions_and_callback(messages, request_id, info, address))

        # “fire-and-forget” background tasks
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)

    def _routing(self, request_id: str, address: str):
        if address is not None:
            return address
        if request_id:
            request_id = request_id.removeprefix("chatcmpl-")
            assert request_id in self.request_id_to_address
            address = self.request_id_to_address.pop(request_id)
        else:
            address = self.weighted_addresses[0][1]
            self.weighted_addresses[0][0] += 1
            heapq.heapreplace(self.weighted_addresses, self.weighted_addresses[0])
        return address

    async def _submit_chat_completions_and_callback(
        self,
        messages: List[Dict[str, str]],
        request_id: str,
        info: Dict[str, Any],
        address: str = None,
    ):
        """Submit chat completion request, wait request finish and do callback."""
        address = self._routing(request_id, address)
        # use new request_id to avoid duplicate request_id problem
        request_id = uuid4().hex
        self.request_id_to_address[request_id] = address

        completions, exception = None, None
        try:
            # NOTE: OpenAI client uses httpx, seems to have performance issue in high concurrency requests.
            completions = await self._chat_completions_aiohttp(
                address,
                messages=messages,
                tools=self.completion_callback.tool_schemas,
                extra_body=self.completion_callback.extra_body,
                extra_headers={"x-request-id": request_id},
                **info["__sampling_params__"],
            )
        except Exception as e:
            # Let user handle the exception
            exception = e

        info["__depth__"] -= 1

        if exception is not None:
            logger.exception(f"chat completion failed with exception: {exception}")
        else:
            try:
                await self.completion_callback(messages, completions, info)
            except Exception as e:
                logger.exception(f"completion callback failed with exception: {e}")

        # No more ongoing completion requests
        if info["__depth__"] == 0:
            info["__done__"].set()

    async def _chat_completions_openai(self, address: str, **chat_complete_request) -> ChatCompletion:
        client = AsyncOpenAI(base_url=f"http://{address}/v1", api_key="token-abc123", timeout=None, max_retries=0)
        return await client.chat.completions.create(**chat_complete_request)

    async def _chat_completions_aiohttp(self, address: str, **chat_complete_request) -> ChatCompletion:
        try:
            extra_body = chat_complete_request.pop("extra_body", {})
            chat_complete_request.update(extra_body or {})
            extra_headers = chat_complete_request.pop("extra_headers")
            timeout = aiohttp.ClientTimeout(total=None)
            session = aiohttp.ClientSession(timeout=timeout)
            async with session.post(
                url=f"http://{address}/v1/chat/completions",
                headers={"Authorization": "Bearer token-abc123", **extra_headers},
                json=chat_complete_request,
            ) as resp:
                data = await resp.json()
                return ChatCompletion(**data)
        finally:
            await session.close()

    async def generate_sequences(self, batch: DataProto) -> DataProto:
        kwargs = dict(
            model=self.model_name,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )

        # override sampling params for validation
        if batch.meta_info.get("validate", False):
            kwargs["top_p"] = self.config.val_kwargs.top_p
            kwargs["temperature"] = self.config.val_kwargs.temperature

        print(f"[ChatCompletionScheduler] generate_sequences sampling params: {kwargs}")

        # NOTE: For multi-turn rollout, repeat raw_prompt n times and process each prompt independently,
        # validation dataset has already been repeated in `PPOTrainer._validate`.
        n = 1 if batch.meta_info.get("validate", False) else self.config.n
        tasks, batch_conversations = [], [None] * len(batch) * n
        for batch_index, conversation in enumerate(batch.non_tensor_batch["raw_prompt"].repeat(n, axis=0)):
            # raw_prompt: [{"role": "user", "content": ""}, ["role": "assistant", "content"], ...]
            batch_conversations[batch_index] = conversation.tolist()

            tasks.append(
                asyncio.create_task(
                    self._submit_chat_completions_semaphore(
                        messages=batch_conversations[batch_index],
                        request_id=None,
                        sampling_params=kwargs,
                    )
                )
            )

        await asyncio.gather(*tasks)
        print("[ChatCompletionScheduler] generate_sequences done")

        return self.completion_callback.postprocess(batch, batch_conversations, n=n)

    async def _submit_chat_completions_semaphore(self, messages: List[Dict[str, str]], request_id: str, sampling_params: Dict[str, Any], address: str = None):
        done = asyncio.Event()

        info = {
            "__done__": done,
            "__depth__": 0,  # indicate how many ongoing completion requests
            "__sampling_params__": sampling_params,
        }

        self.submit_chat_completions(messages=messages, request_id=request_id, info=info, address=address)

        # Wait until all completion requests are done
        await done.wait()


class MicroBatchScheduler(ChatCompletionScheduler):
    def __init__(self, config, model_path, server_addresses, max_cache_size=10000, max_inflight_req=8, external_calls_plugins: List[AsyncCallbackMixin] = None):
        super().__init__(config, model_path, server_addresses, max_cache_size)
        self._validate_callback()
        self.max_inflight_req = max_inflight_req
        self.server_addresses = server_addresses
        self.number_of_servers = len(server_addresses)
        self.external_calls_plugins = external_calls_plugins
        self._init_global_resource()
        self.engine_call_actors = self._init_engine_call_actors(server_address=server_addresses, send_queue=self.send_queue, reduce_queue=self.reduce_queue)

    def _validate_callback(self):
        if self.completion_callback is None:
            raise ValueError("completion_callback is None")
        if not isinstance(self.completion_callback, AsyncCallbackMixin):
            raise ValueError("completion_callback mixin AsyncCallbackMixin")

    def _init_reduce_thread(self):
        self.reduce_thread = threading.Thread(target=self.handle_reduce_req, args=(self.reduce_data_queue,), daemon=True, name="reduce_data_thread")
        self.reduce_thread.start()

    def _init_global_resource(self):
        self.loop = asyncio.get_event_loop()
        self.global_data_queue = asyncio.Queue()
        self.local_data_queue_group = QueueGroup(self.number_of_servers, [asyncio.Queue() for _ in range(self.number_of_servers)])
        self.tool_req_queue = asyncio.Queue()
        self.reduce_data_queue = asyncio.Queue()

    def _init_engine_call_actors(self, server_address, max_inflight_req):
        # we use a group of coroutine to consume send_queue and produce reduce_queue
        # since the asyncio.Queue is not thread safe.
        # max_inflight_req consumer coroutine to get element from local_queue and submit to vllm
        actors = []
        for idx, addr in enumerate(server_address):
            for _ in range(max_inflight_req):
                work_fn = functools.partial(self.handle_rollout_req, addr, self.tool_req_queue, self.reduce_data_queue)
                actor = WorkStealingActor(worker_id=idx, local_queues=self.local_data_queue_group, global_queue=self.global_data_queue, work_fn=work_fn)
                actors.append(actor)
                asyncio.create_task(actor.run())
        return actors

    async def handle_rollout_req(self, addr, reduce_queue: asyncio.Queue, external_call: AsyncCallbackMixin, actor_meta: ActorMeta, rollout_req: RolloutReq):
        logger.debug(f"[MicroBatchChatCompletionScheduler] _consumer process get sample, submit to engine {addr}")
        request_id = uuid4().hex
        completions, exception = None, None
        messages = rollout_req.messages
        try:
            # NOTE: OpenAI client uses httpx, seems to have performance issue in high concurrency requests.
            completions = await self._chat_completions_aiohttp(
                address=addr,
                messages=messages,
                tools=rollout_req.tools_schema,
                extra_body=rollout_req.extra_body,
                extra_headers={"x-request-id": request_id},
                **rollout_req.sampling_params,
            )
        except Exception as e:
            # Let user handle the exception
            exception = e
        message = completions.choices[0].message.model_dump(exclude_unset=True, exclude_none=True)
        if "content" not in message:
            message["content"] = ""
        messages.append(message)
        resp = RolloutResp(completions=completions, info=rollout_req.info, exception=exception, messages=messages)
        if external_call.hit(resp):
            external_call.put(CallsReq(rollout_resp=resp, actor_meta=actor_meta))
        else:
            reduce_queue.put(resp)
        logger.debug("[MicroBatchChatCompletionScheduler] _consumer process done")

    # maybe we can make this sink_queue as a pubsub proxy using zmq
    async def handle_reduce_req(self, batch_size, sink_queue: asyncio.Queue = None):
        batch_conversations = [None] * batch_size
        counter = 0
        while counter < batch_size:
            sample: RolloutResp = await self.reduce_data_queue.get()
            logger.debug(f"[MicroBatchChatCompletionScheduler] _gather_result counter: {counter},sample: {sample}")
            if sink_queue is not None:
                sink_queue.put(sample)
            counter += 1
            if sample.exception is not None:
                # assert exception is None, f"exception: {exception}"
                raise sample.exception
            conversation, batch_index = (
                sample.info["conversation"],
                sample.info["batch_index"],
            )
            conversations = []
            for choice in sample.completions.choices:
                chat = conversation.copy()
                chat.append({"role": choice.message.role, "content": choice.message.content})
                conversations.append(chat)
            batch_conversations[batch_index] = conversations
        logger.debug("[MicroBatchChatCompletionScheduler] _gather_result done for one batch")
        return batch_conversations

    async def generate_sequences(self, batch: DataProto, **sampling_params) -> DataProto:
        kwargs = dict(
            model=self.model_name,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )

        # override sampling params for validation
        if batch.meta_info.get("validate", False):
            kwargs["top_p"] = self.config.val_kwargs.top_p
            kwargs["temperature"] = self.config.val_kwargs.temperature

        logger.info(f"[ChatCompletionScheduler] generate_sequences sampling params: {kwargs}")

        # NOTE: For multi-turn rollout, repeat raw_prompt n times and process each prompt independently,
        # validation dataset has already been repeated in `PPOTrainer._validate`.
        n = 1 if batch.meta_info.get("validate", False) else self.config.n
        batch_conversations = [None] * len(batch) * n
        for batch_index, conversation in enumerate(batch.non_tensor_batch["raw_prompt"].repeat(n, axis=0)):
            # raw_prompt: [{"role": "user", "content": ""}, ["role": "assistant", "content"], ...]
            batch_conversations[batch_index] = conversation.tolist()

            await self.send_queue.put(
                RolloutReq(
                    completions=None,
                    info={
                        "batch_index": batch_index,
                        "conversation": list(conversation),
                    },
                    messages=conversation.tolist(),
                    model_name=self.model_name,
                    chat_complete_request=kwargs,
                    exceptoin=None,
                )
            )
        batch_conversations = await self._gather_result(len(batch))
        logger.info("[MicroBatchChatCompletionScheduler] generate_sequences done")
        return self.completion_callback.postprocess(batch, batch_conversations, n=n)
