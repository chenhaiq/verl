import asyncio
import json
import os
from functools import wraps
from typing import Dict, List
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
from datasets import load_dataset
from omegaconf import OmegaConf
from openai.types.chat.chat_completion import ChatCompletion, ChatCompletionMessage, Choice

from verl.protocol import DataProto
from verl.workers.rollout.chat_scheduler.apis import CallsReq, CoroExternalCallsPlugin, RolloutReq, RolloutResp
from verl.workers.rollout.chat_scheduler.chat_scheduler import MicroBatchScheduler, ToolCompletionCallback

do_bench = True


def skip_if_false(exp):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not exp:
                pytest.skip("test will be skipped")

        return wrapper

    return decorator


class NoHitCompletionCallback(ToolCompletionCallback, CoroExternalCallsPlugin):
    def __init__(self, config, scheduler):
        ToolCompletionCallback.__init__(self, config, scheduler)
        CoroExternalCallsPlugin.__init__(self, num_workers=2)
        self.req_counter = {}

    def hit(self, req: CallsReq):
        return False

    def __call__(self, req: CallsReq):
        raise ValueError("NoHitCompletionCallback")

    def postprocess(self, batch: DataProto, batch_conversations: List[List[Dict[str, str]]], n: int):
        return batch_conversations


class ThirdTurnCallback(NoHitCompletionCallback):
    def hit(self, req: RolloutResp):
        session_id = req.request.verl_session_id
        if session_id not in self.req_counter.keys():
            self.req_counter[session_id] = 1
            return True
        else:
            self.req_counter[session_id] += 1
            if self.req_counter[session_id] > 3:
                return False
            else:
                return True

    def __call__(self, req):
        msg = req.rollout_resp.request.messages
        resp_str = json.dumps(
            {
                "local_id": req.actor_meta.local_id,
                "actor_id": req.actor_meta.actor_id,
                "turns": self.req_counter[req.rollout_resp.request.verl_session_id],
            },
        )
        msg.append(ChatCompletionMessage(role="assistant", content=resp_str))
        rollout_req = RolloutReq(
            verl_session_id=req.rollout_resp.request.verl_session_id,
            model_name=req.rollout_resp.request.model_name,
            sampling_params=req.rollout_resp.request.sampling_params,
            tools_schema=req.rollout_resp.request.tools_schema,
            extra_body=req.rollout_resp.request.extra_body,
            messages=msg,
        )
        return rollout_req


class TestMicroBatchScheduler:
    @pytest.fixture
    def ray_env(self):
        runtime_env = {
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "DEBUG",
                "VLLM_USE_V1": "1",
            }
        }
        return runtime_env

    @pytest.fixture
    def aime_dataset(self):
        dataset = load_dataset("Maxwell-Jia/AIME_2024", split="train")
        prompts = DataProto(non_tensor_batch={"raw_prompt": np.array([[{"role": "user", "content": problem}] for problem in dataset["Problem"]])})
        return [prompts, dataset]

    @pytest.fixture
    def code_dataset(self):
        def gen_code():
            dataset = load_dataset("/demo-huabei2/chenhaiquan/dataset/Eurus-2-RL-Data/", split="train")
            print("finish")
            prompts = DataProto(non_tensor_batch={"raw_prompt": np.array([dataset[idx]["prompt"] for idx in range(256, 256 + 8192)])})
            return [prompts, dataset]

        return gen_code

    @pytest.fixture
    def small_model_path(self):
        return "Qwen/Qwen3-4B"

    @pytest.fixture
    def large_model_path(self):
        return "/demo-huabei2/common-models/Qwen/Qwen2.5-7B-Instruct"

    @pytest.fixture
    def sampling_params(self):
        return dict(
            n=1,
            temperature=0,
            top_p=1,
            top_k=-1,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            repetition_penalty=1.0,
            skip_special_tokens=True,
            spaces_between_special_tokens=True,
            ignore_eos=False,
        )

    @pytest.fixture
    def chat_completion(self):
        return ChatCompletion(
            id="40359df5b352b344bc8e",
            choices=[
                Choice(
                    finish_reason="stop",
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content="The answer is 42.",
                    ),
                )
            ],
            created=1690917600,
            model="gpt-3.5-turbo",
            object="chat.completion",
        )

    @patch("verl.workers.rollout.chat_scheduler.requests.chat_completions_aiohttp", new_callable=AsyncMock)
    def test_global_queue_put(self, mock_chat_completions_aiohttp: AsyncMock, ray_env, aime_dataset, small_model_path, chat_completion):
        os.environ["VERL_QUEUE_LOGGING_LEVEL"] = "DEBUG"
        os.environ["VERL_LOGGING_LEVEL"] = "DEBUG"
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def run_test():
            # Load config
            mock_chat_completions_aiohttp.return_value = chat_completion
            config = OmegaConf.load("verl/trainer/config/ppo_trainer.yaml")
            config.actor_rollout_ref.model.path = small_model_path
            config.actor_rollout_ref.rollout.chat_scheduler.name = "micro_batch_scheduler"
            config.actor_rollout_ref.rollout.multi_turn.completion_callback = "tests.workers.rollout.test_vllm_micro_batch_scheduler.NoHitCompletionCallback"
            prompts, _ = aime_dataset[0], aime_dataset[1]
            print("length of data proto : ", len(prompts))
            # Init sandbox and async rollout manager
            scheduler = MicroBatchScheduler(config, server_addresses=[1, 2, 3, 4], max_inflight_req=2, enable_work_stealing=False)

            re = await scheduler.generate_sequences(batch=prompts)
            assert len(re) == len(prompts), "length of re is not equal to length of prompts,re is {} and prompts is {}".format(len(re), len(prompts))
            await scheduler.shut_down_actors()

        loop.run_until_complete(run_test())
        loop.stop()
        loop.close()

    @patch("verl.workers.rollout.chat_scheduler.requests.chat_completions_aiohttp", new_callable=AsyncMock)
    def test_local_queue_put(self, mock_chat_completions_aiohttp: AsyncMock, ray_env, aime_dataset, small_model_path, chat_completion):
        os.environ["VERL_QUEUE_LOGGING_LEVEL"] = "INFO"
        os.environ["VERL_LOGGING_LEVEL"] = "DEBUG"
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def run_test():
            # Load config
            mock_chat_completions_aiohttp.return_value = chat_completion
            config = OmegaConf.load("verl/trainer/config/ppo_trainer.yaml")
            config.actor_rollout_ref.model.path = small_model_path
            config.actor_rollout_ref.rollout.chat_scheduler.name = "micro_batch_scheduler"
            config.actor_rollout_ref.rollout.multi_turn.completion_callback = "tests.workers.rollout.test_vllm_micro_batch_scheduler.ThirdTurnCallback"
            prompts, _ = aime_dataset[0], aime_dataset[1]
            print("length of data proto : ", len(prompts))
            # Init sandbox and async rollout manager
            scheduler = MicroBatchScheduler(config, server_addresses=[1, 2, 3, 4], max_inflight_req=2, enable_work_stealing=False)

            re = await scheduler.generate_sequences(batch=prompts)
            assert len(re) == len(prompts), "length of re is not equal to length of prompts,re is {} and prompts is {}".format(len(re), len(prompts))
            await scheduler.shut_down_actors()
            actor_id_count = {}
            for sample in re:
                print(sample)
                assert len(sample) == 2 + 2 * 3, len(sample)
                turns_msg = [json.loads(sample[2].content), json.loads(sample[4].content), json.loads(sample[6].content)]
                actor_id = turns_msg[0]["actor_id"]
                if actor_id not in actor_id_count:
                    actor_id_count[actor_id] = 0
                actor_id_count[actor_id] += 1
                for real_id in turns_msg:
                    assert real_id["actor_id"] == actor_id, f"local queue mismatch: {real_id}, {actor_id}"
            values = actor_id_count.values()
            assert len(values) == 4, "actor id count is not equal to 4"

        loop.run_until_complete(run_test())
        loop.stop()
        loop.close()

    # def test_micro_batch_scheduler(self, ray_env, aime_dataset, small_model_path):
    #     ray.init(
    #         runtime_env=ray_env,
    #     )
    #     # Load config
    #     config = OmegaConf.load("verl/trainer/config/ppo_trainer.yaml")
    #     config.actor_rollout_ref.model.path = small_model_path
    #     config.actor_rollout_ref.rollout.mode = "async"
    #     config.actor_rollout_ref.rollout.chat_scheduler = "examples.ppo_trainer.naive_chat_scheduler.MicroBatchChatCompletionScheduler"
    #     config.actor_rollout_ref.rollout.prompt_length = 8192
    #     config.actor_rollout_ref.rollout.response_length = 8192
    #     config.actor_rollout_ref.rollout.temperature = 0.0
    #     config.actor_rollout_ref.rollout.repetition_penalty = 1.0

    #     # Init sandbox and async rollout manager
    #     async_rollout_manager = init_async_rollout_manager(config, scheduler_kwargs={"max_inflight_req": 4})

    #     # Build dataset
    #     prompts, dataset = aime_dataset[0], aime_dataset[1]
    #     print(f"length of data proto : {len(prompts)}")
    #     start_time = time.time()
    #     micro_result = async_rollout_manager.generate_sequences(prompts=prompts)
    #     print(f"length of micro_result : {len(micro_result)}")
    #     print(f"time cost for micro_result : {time.time() - start_time}")
    #     torch.save(micro_result, "micro_result.pt")
    #     assert len(micro_result) == len(dataset)
    #     ray.timeline("micro_batch_scheduler.json")
    #     ray.shutdown()

    #     ray.init(
    #         runtime_env=ray_env,
    #     )

    #     config.actor_rollout_ref.rollout.chat_scheduler = "examples.ppo_trainer.naive_chat_scheduler.NaiveChatCompletionScheduler"

    #     # Init sandbox and async rollout manager
    #     async_rollout_manager = init_async_rollout_manager(config)
    #     start_time = time.time()
    #     native_result = async_rollout_manager.generate_sequences(prompts=prompts)
    #     print(f"length of native_result : {len(native_result)}")
    #     print(f"time cost for native_result : {time.time() - start_time}")
    #     torch.save(native_result, "native_result.pt")
    #     assert len(native_result) == len(dataset)
    #     ray.timeline("native_batch_scheduler.json")
    #     for i in range(len(dataset)):
    #         """
    #         a[0].__dict__['batch']
    #         TensorDict(
    #             fields={
    #                 attention_mask: Tensor(shape=torch.Size([8561]), device=cpu, dtype=torch.int64, is_shared=False),
    #                 input_ids: Tensor(shape=torch.Size([8561]), device=cpu, dtype=torch.int64, is_shared=False),
    #                 position_ids: Tensor(shape=torch.Size([8561]), device=cpu, dtype=torch.int64, is_shared=False),
    #                 prompts: Tensor(shape=torch.Size([363]), device=cpu, dtype=torch.int64, is_shared=False),
    #                 responses: Tensor(shape=torch.Size([8198]), device=cpu, dtype=torch.int64, is_shared=False)},
    #             batch_size=torch.Size([]),
    #             device=None,
    #             is_shared=False)
    #         """
    #         torch.allclose(micro_result[i].__dict__["batch"]["responses"], native_result[i].__dict__["batch"]["responses"])
    #         torch.allclose(micro_result[i].__dict__["batch"]["attention_mask"], native_result[i].__dict__["batch"]["attention_mask"])
    #         torch.allclose(micro_result[i].__dict__["batch"]["input_ids"], native_result[i].__dict__["batch"]["input_ids"])
    #         torch.allclose(micro_result[i].__dict__["batch"]["position_ids"], native_result[i].__dict__["batch"]["position_ids"])
    #         torch.allclose(micro_result[i].__dict__["batch"]["prompts"], native_result[i].__dict__["batch"]["prompts"])

    # # @skip_if_false(True)
    # def test_bench_micro_batch_scheduler(self, ray_env, code_dataset, sampling_params, large_model_path):
    #     if not do_bench:
    #         return
    #     ray.init(
    #         runtime_env=ray_env,
    #     )
    #     # Load config
    #     config = OmegaConf.load("verl/trainer/config/ppo_trainer.yaml")
    #     config.actor_rollout_ref.model.path = large_model_path
    #     config.actor_rollout_ref.rollout.mode = "async"
    #     config.actor_rollout_ref.rollout.chat_scheduler = "examples.ppo_trainer.naive_chat_scheduler.MicroBatchChatCompletionScheduler"
    #     config.actor_rollout_ref.rollout.prompt_length = 4096
    #     config.actor_rollout_ref.rollout.response_length = 4096
    #     config.actor_rollout_ref.rollout.temperature = 0.1
    #     config.actor_rollout_ref.rollout.repetition_penalty = 1.0
    #     config.actor_rollout_ref.rollout.gpu_memory_utilization = 0.5
    #     config.actor_rollout_ref.rollout.preemption_mode = "swap"
    #     config.actor_rollout_ref.rollout.tensor_model_parallel_size = 1

    #     # Init sandbox and async rollout manager
    #     async_rollout_manager = init_async_rollout_manager(config, scheduler_kwargs={"max_inflight_req": 384})
    #     # Build dataset
    #     prompts, dataset = code_dataset()
    #     print(f"length of data proto : {len(prompts)}")
    #     start_time = time.time()
    #     micro_result = async_rollout_manager.generate_sequences(prompts=prompts, **sampling_params)
    #     print(f"length of micro_result : {len(micro_result)}")
    #     print(f"time cost for micro_result : {time.time() - start_time}")
    #     torch.save(micro_result, "1024-micro_result.pt")
    #     assert len(micro_result) == len(prompts)
    #     ray.timeline("debug-1024-micro_batch_scheduler.json")
    #     ray.shutdown()

    #     ray.init(
    #         runtime_env=ray_env,
    #     )

    #     config.actor_rollout_ref.rollout.chat_scheduler = "examples.ppo_trainer.naive_chat_scheduler.NaiveChatCompletionScheduler"

    #     # Init sandbox and async rollout manager
    #     async_rollout_manager = init_async_rollout_manager(config)
    #     start_time = time.time()
    #     native_result = async_rollout_manager.generate_sequences(prompts=prompts, **sampling_params)
    #     print(f"length of native_result : {len(native_result)}")
    #     print(f"time cost for native_result : {time.time() - start_time}")
    #     torch.save(native_result, "1024-native_result.pt")
    #     assert len(native_result) == len(prompts)
    #     ray.timeline("debug-1024-native_batch_scheduler.json")
