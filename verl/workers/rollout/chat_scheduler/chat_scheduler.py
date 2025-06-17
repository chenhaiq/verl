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
import itertools
import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from uuid import uuid4

import numpy as np
import torch
from cachetools import LRUCache
from omegaconf import DictConfig
from openai.types.chat.chat_completion import ChatCompletion
from tensordict import TensorDict
from transformers import PreTrainedTokenizer

from verl.protocol import DataProto
from verl.tools.base_tool import initialize_tools_from_config
from verl.utils.fs import copy_to_local
from verl.utils.tokenizer import hf_tokenizer
from verl.workers.rollout.chat_scheduler.apis import AsyncCallbackMixin, CallsReq, CoroExternalCallsPlugin, RolloutReq, RolloutResp
from verl.workers.rollout.chat_scheduler.utils import ActorMeta, QueueGroup, WorkStealingActor

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


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
        self.tokenizer: PreTrainedTokenizer = hf_tokenizer(local_path, trust_remote_code=True)

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


class ToolCompletionCallback(CompletionCallback):
    def __init__(self, config: DictConfig, scheduler: "ChatCompletionScheduler"):
        super().__init__(config, scheduler)

        # TODO: add reward manager to calculate reward score once a sample finish

    async def __call__(self, messages: List[Dict[str, str]], completions: ChatCompletion, info: Dict[str, Any]):
        message = completions.choices[0].message.model_dump(exclude_unset=True, exclude_none=True)
        if "content" not in message:
            message["content"] = ""
        messages.append(message)
        finish_reason = completions.choices[0].finish_reason

        # STEP 0: check if we reach max turns
        if self.max_turns and len(messages) >= self.max_turns:
            print(f"[id={completions.id},turn={len(messages)},finish_reason={finish_reason}] Reach max turns, done!")
            return

        # STEP 1: check if the model called tools
        if finish_reason != "tool_calls":
            print(f"[id={completions.id},turn={len(messages)},finish_reason={finish_reason}] No tool called, done!")
            return

        # STEP 2: call tools
        tool_calls = completions.choices[0].message.tool_calls
        print(f"[id={completions.id},turn={len(messages)},finish_reason={finish_reason}] Call {len(tool_calls)} tools")
        tasks = []
        for tool_call in tool_calls:
            tasks.append(self._call_tool(tool_call))
        tool_responses = await asyncio.gather(*tasks)
        if any(isinstance(item, Exception) for item in tool_responses):
            logger.debug(f"[id={completions.id},turn={len(messages)},finish_reason={finish_reason}] Error when calling tools, done!")
            return
        messages.extend(tool_responses)

        # STEP 3: resubmit completion request with tool responses
        self.scheduler.submit_chat_completions(messages=messages, request_id=completions.id, info=info)

    async def _call_tool(self, tool_call) -> Dict[str, str]:
        """Call tool and return tool response."""
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)
        tool = self.tools[tool_name]

        instance_id = await tool.create()
        try:
            tool_response, tool_reward_score, tool_metrics = await tool.execute(instance_id, tool_args)
        except Exception as e:
            logger.exception(f"Error when executing tool: {e}")
            return e
        finally:
            await tool.release(instance_id)

        return {
            "role": "tool",
            "content": tool_response,
            "tool_call_id": tool_call.id,
        }

    def postprocess(self, batch: DataProto, batch_conversations: List[List[Dict[str, str]]], n: int) -> DataProto:
        # NOTE: consistent with batch version of generate_sequences in vllm_rollout_spmd.py
        # prompts: left pad
        # responses: right pad
        # input_ids: prompt + response
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]

        # prompts: [prompt] from input dataset
        prompts = [self.tokenizer.apply_chat_template(prompt, tools=self.tool_schemas, add_generation_prompt=True, tokenize=False) for prompt in batch.non_tensor_batch["raw_prompt"]]
        assert len(batch_conversations) == len(prompts) * n

        # sequences: [prompt + response]
        sequences = [self.tokenizer.apply_chat_template(conversation, tools=self.tool_schemas, add_generation_prompt=False, tokenize=False) for conversation in batch_conversations]

        # responses: [response]
        responses = [sequence[len(prompts[i // n]) :] for i, sequence in enumerate(sequences)]

        prompts = self.tokenizer(prompts, return_tensors="pt", padding="longest", padding_side="left")
        responses = self.tokenizer(responses, return_tensors="pt", padding="longest", padding_side="right")
        if n > 1:
            prompts["input_ids"] = prompts["input_ids"].repeat_interleave(n, dim=0)
            prompts["attention_mask"] = prompts["attention_mask"].repeat_interleave(n, dim=0)

        # response_mask: response mask with tools calling masked out
        response_mask = self._mask_out_tools_calling_tokens(batch.non_tensor_batch["raw_prompt"].repeat(n, axis=0), batch_conversations, responses["input_ids"], responses["attention_mask"])

        input_ids = torch.cat([prompts["input_ids"], responses["input_ids"]], dim=1)
        attention_mask = torch.cat([prompts["attention_mask"], responses["attention_mask"]], dim=1)
        position_ids = (attention_mask.cumsum(dim=1) - 1) * attention_mask

        batch = TensorDict(
            {
                "prompts": prompts["input_ids"],  # [bsz, prompt_length]
                "responses": responses["input_ids"],  # [bsz, response_length]
                "response_mask": response_mask,  # [bsz, response_length]
                "input_ids": input_ids,  # [bsz, prompt_length + response_length]
                "attention_mask": attention_mask,  # [bsz, prompt_length + response_length]
                "position_ids": position_ids,  # [bsz, prompt_length + response_length]
            },
            batch_size=len(input_ids),
        )

        num_turns = np.array([len(conversation) for conversation in batch_conversations], dtype=np.int32)
        return DataProto(batch=batch, non_tensor_batch={"__num_turns__": num_turns})

    def _mask_out_tools_calling_tokens(
        self,
        raw_prompts: List[List[Dict[str, str]]],
        batch_conversations: List[List[Dict[str, str]]],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Mask out tools calling tokens in the responses.

        Args:
            raw_prompts: [prompt] from input dataset
            batch_conversations: [prompt + response]
            input_ids: responses tokens
            attention_mask: responses attention mask

        Returns:
            mask: (batch_size, response_length)
        """
        batch_size = input_ids.size(0)
        assert len(raw_prompts) == batch_size, f"{len(raw_prompts)} != {batch_size}"
        assert len(batch_conversations) == batch_size, f"{len(batch_conversations)} != {batch_size}"

        # Deduplicate adjacent tool calls, since they're merged into one turn.
        # [user, assistant, tool, tool, assistant] -> [user, assistant, tool, assistant]
        # TODO: it's chat_template specific, find a more generic way to do this.
        def deduplicate_adjacent_tool_calls(roles):
            result = []
            for role, group in itertools.groupby(roles):
                if role == "tool":
                    result.append(role)
                else:
                    result.extend(group)
            return result

        loss_mask = attention_mask.clone()
        for i in range(batch_size):
            responses = batch_conversations[i][len(raw_prompts[i]) :]
            assert len(responses) > 0, f"responses is empty: {responses}"

            roles = deduplicate_adjacent_tool_calls([response["role"] for response in responses])
            # Each turn should be: [BOS]...[EOS]
            eos_indices = input_ids[i].eq(self.tokenizer.eos_token_id).nonzero().squeeze(1)[: len(roles)]
            for j in range(len(roles)):
                if roles[j] == "tool":
                    bos = eos_indices[j - 1] + 1 if j > 0 else 0
                    eos = eos_indices[j]
                    loss_mask[i, bos : eos + 1] = 0

        return loss_mask


class AsyncToolCompletionCallback(ToolCompletionCallback, CoroExternalCallsPlugin):
    def __init__(self, config: DictConfig, scheduler: "ChatCompletionScheduler"):
        print("init_async tools")
        ToolCompletionCallback.__init__(self, config, scheduler)
        CoroExternalCallsPlugin.__init__(self, num_workers=5)

    def hit(self, req: CallsReq):
        completions = req.completions
        messages = req.messages
        finish_reason = completions.choices[0].finish_reason

        # STEP 0: check if we reach max turns
        if self.max_turns and len(messages) >= self.max_turns:
            logger.debug(f"[id={completions.id},turn={len(messages)},finish_reason={finish_reason}] Reach max turns, done!")
            return True

        # STEP 1: check if the model called tools
        if finish_reason != "tool_calls":
            logger.debug(f"[id={completions.id},turn={len(messages)},finish_reason={finish_reason}] No tool called, done!")
            return False

    async def __call__(self, req: CallsReq):
        completions = req.rollout_resp.completions
        messages = req.rollout_resp.messages
        finish_reason = completions.choices[0].finish_reason

        # call tools
        tool_calls = completions.choices[0].message.tool_calls
        logger.debug(f"[id={completions.id},turn={len(messages)},finish_reason={finish_reason}] Call {len(tool_calls)} tools")
        tasks = []
        for tool_call in tool_calls:
            tasks.append(self._call_tool(tool_call))
        tool_responses = await asyncio.gather(*tasks)
        if any(isinstance(item, Exception) for item in tool_responses):
            logger.debug(f"[id={completions.id},turn={len(messages)},finish_reason={finish_reason}] Error when calling tools, done!")
            return
        messages.extend(tool_responses)

        # STEP 3: send it back to local_queue
        new_rollout_req = RolloutReq(
            info=req.rollout_resp.info,
            messages=messages,
            model_name=req.rollout_resp.model_name,
            chat_complete_request=req.rollout_resp.chat_complete_request,
        )
        req.actor_meta.queue_group.push(req.actor_meta.actor_id, new_rollout_req)


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
        from verl.workers.rollout.chat_scheduler.requests import chat_completions_openai

        return await chat_completions_openai(address, **chat_complete_request)

    async def _chat_completions_aiohttp(self, address: str, **chat_complete_request) -> ChatCompletion:
        from verl.workers.rollout.chat_scheduler.requests import chat_completions_aiohttp

        return await chat_completions_aiohttp(address, **chat_complete_request)

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
    def __init__(self, config, server_addresses, max_cache_size=10000, rollout_rate=1, max_inflight_req=8, rollout_req_handler=None, reduce_handler=None, enable_work_stealing=True):
        super().__init__(config, server_addresses, max_cache_size)
        self._validate_callback()
        self.max_inflight_req = max_inflight_req
        self.server_addresses = server_addresses
        self.enable_work_stealing = enable_work_stealing
        self.number_of_servers = len(server_addresses)
        self.rollout_rate = rollout_rate
        self.rollout_req_handler = rollout_req_handler if rollout_req_handler else self.default_handle_rollout_req
        self.reduce_handler = reduce_handler if reduce_handler else self.default_handle_reduce_req
        self._init_global_resource()
        # TODO better implement a supervisor-tree pattern, include dead-letter-queue to monitor whether any actor exit unexpectly
        self.engine_call_actors: List[WorkStealingActor] = self._init_engine_call_actors(server_address=server_addresses, max_inflight_req=max_inflight_req)
        self.wake_up_engine_actor()

    def set_rollout_rate(self, rate):
        assert rate <= 1 and rate > 0, "rollout rate must be in (0, 1]"
        self.rollout_rate = rate

    def _get_rollout_batch_size(self, data_batch_size):
        return int(data_batch_size * self.rollout_rate)

    def _validate_callback(self):
        if self.completion_callback is None:
            raise ValueError("completion_callback is None")
        if not isinstance(self.completion_callback, AsyncCallbackMixin):
            raise ValueError("completion_callback mixin AsyncCallbackMixin")
        logger.error(f"completion_callback: {self.completion_callback}")

    def _init_global_resource(self):
        # TODO use ZMQ to implement pub-sub for debug purpose
        self.loop = asyncio.get_event_loop()
        self.global_data_queue = asyncio.Queue()
        self.local_data_queue_group = QueueGroup(self.number_of_servers, [asyncio.Queue() for _ in range(self.number_of_servers)])
        self.reduce_data_queue = asyncio.Queue()

    def _init_engine_call_actors(self, server_address, max_inflight_req):
        # we use a group of coroutine to consume send_queue and produce reduce_queue
        # since the asyncio.Queue is not thread safe.
        # max_inflight_req consumer coroutine to get element from local_queue and submit to vllm
        actors = []
        counter = 0
        for idx, addr in enumerate(server_address):
            print(f"[MicroBatchChatCompletionScheduler] init engine call actor {addr}, max_inflight_req: {max_inflight_req}")
            for _ in range(max_inflight_req):
                work_fn = functools.partial(
                    self.rollout_req_handler,
                    addr,
                    self.reduce_data_queue,
                    self.completion_callback,
                )
                actor = WorkStealingActor(worker_id=idx, local_id=counter, local_queues=self.local_data_queue_group, global_queue=self.global_data_queue, work_fn=work_fn, enable_work_stealing=self.enable_work_stealing)
                actors.append(actor)
                counter += 1
        print(f"[MicroBatchChatCompletionScheduler] init engine call actors done, total: {len(actors)}")
        return actors

    async def cancel_all_req(self):
        evts = []
        for actor in self.engine_call_actors:
            maybe_set_evt: asyncio.Event = actor.cancel_task()
            if not maybe_set_evt.is_set():
                evts.append(maybe_set_evt.wait())
        print(f"cancel req with length: {len(evts)}")
        await asyncio.gather(*evts)

    def wake_up_engine_actor(self):
        for actor in self.engine_call_actors:
            actor.wakeup()

    async def shut_down_actors(self):
        print("shut down engine actors with length: ", len(self.engine_call_actors))
        for actor in self.engine_call_actors:
            print("ready to shutdown actor: ", actor.actor_meta)
            await actor.shutdown()
        print("[MicroBatchChatCompletionScheduler] shut down engine actor")
        await self.completion_callback.shutdown()
        print("[MicroBatchChatCompletionScheduler] shut down completion callback")

    async def default_handle_rollout_req(self, addr, reduce_queue: asyncio.Queue, external_call: AsyncCallbackMixin, actor_meta: ActorMeta, rollout_req: RolloutReq):
        from verl.workers.rollout.chat_scheduler.requests import chat_completions_aiohttp

        print(f"[MicroBatchChatCompletionScheduler] _consumer process get sample, addr: {addr}, actor_meta: {actor_meta}")
        request_id = uuid4().hex
        completions, exception, message = None, None, {}
        messages = rollout_req.messages
        try:
            # NOTE: OpenAI client uses httpx, seems to have performance issue in high concurrency requests.
            print(f"[MicroBatchChatCompletionScheduler] _consumer process get sample, submit to engine {addr}")
            completions = await chat_completions_aiohttp(
                address=addr,
                messages=messages,
                tools=rollout_req.tools_schema,
                extra_body=rollout_req.extra_body,
                extra_headers={"x-request-id": request_id},
                **rollout_req.sampling_params,
            )
            message = completions.choices[0].message.model_dump(exclude_unset=True, exclude_none=True)
        except Exception as e:
            print(f"chat completion failed with exception: {e}")
            exception = e
        print(f"[MicroBatchChatCompletionScheduler] _consumer process get sample done,meesage: {message}", actor_meta)
        if "content" not in message:
            message["content"] = ""
        messages.append(message)
        resp = RolloutResp(request=rollout_req, completions=completions, exception=exception, req_id=request_id, messages=messages)
        try:
            if external_call.hit(resp):
                print(f"[id={completions.id},turn={len(messages)},finish_reason={completions.choices[0].finish_reason}] Call tools")
                external_call.put(CallsReq(rollout_resp=resp, actor_meta=actor_meta))
            else:
                print(f"[MicroBatchChatCompletionScheduler] _consumer process put sample to reduce_queue,idx: {actor_meta.actor_id}")
                reduce_queue.put_nowait(resp)
        except Exception as e:
            print(f"[MicroBatchChatCompletionScheduler] _consumer process put sample to reduce_queue failed,idx: {actor_meta.actor_id}, exception: {e}")
            resp.exception = e
            reduce_queue.put_nowait(resp)
        print("[MicroBatchChatCompletionScheduler] _consumer process done")

    # maybe we can make this sink_queue as a pubsub proxy using zmq
    async def default_handle_reduce_req(self, batch_size, sink_queue: asyncio.Queue = None):
        batch_conversations = [None] * batch_size
        counter = 0
        while counter < batch_size:
            print(f"[MicroBatchChatCompletionScheduler] _gather_result counter: {counter}")
            sample: RolloutResp = await self.reduce_data_queue.get()
            if sink_queue is not None:
                sink_queue.put(sample)
            if sample.exception is not None:
                # assert exception is None, f"exception: {exception}"
                raise sample.exception
            batch_conversations[counter] = sample.messages
            counter += 1
        print("[MicroBatchChatCompletionScheduler] _gather_result done for one batch")
        return batch_conversations  # -》 当前要训的+不要训的/部分完成的。

    async def generate_sequences(self, batch: DataProto) -> DataProto:
        self.wake_up_engine_actor()
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
        batch_conversations = [None] * len(batch) * n
        for batch_index, conversation in enumerate(batch.non_tensor_batch["raw_prompt"].repeat(n, axis=0)):
            # raw_prompt: [{"role": "user", "content": ""}, ["role": "assistant", "content"], ...]
            batch_conversations[batch_index] = conversation.tolist()

            self.global_data_queue.put_nowait(
                RolloutReq(
                    messages=conversation.tolist(),
                    model_name=self.model_name,
                    sampling_params=kwargs,
                    tools_schema=self.completion_callback.tool_schemas,
                    extra_body=self.completion_callback.extra_body,
                    verl_session_id=uuid4().hex,
                )
            )
        print("[MicroBatchChatCompletionScheduler] generate_sequences start, with len(batch): ", len(batch))
        batch_conversations = await self.reduce_handler(self._get_rollout_batch_size(len(batch)))
        print(f"partial rollout done, cancel all left request, real size: {len(batch_conversations)}")
        await self.cancel_all_req()
        print("[MicroBatchChatCompletionScheduler] generate_sequences done")
        return self.completion_callback.postprocess(batch, batch_conversations, n=n)
