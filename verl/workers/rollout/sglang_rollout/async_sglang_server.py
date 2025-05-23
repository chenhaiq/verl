# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
import logging

import ray
from omegaconf import DictConfig
from starlette.requests import Request
from starlette.responses import JSONResponse

from verl.workers.rollout.async_server import AsyncServerBase

logger = logging.getLogger(__file__)


@ray.remote(num_cpus=1)
class AsyncSglangServer(AsyncServerBase):
    def __init__(self, config: DictConfig, dp_size: int, dp_rank: int, wg_prefix: str):
        """
        Args:
            config: 角色展开配置（actor_rollout_ref config）
            wg_prefix: 工作进程组前缀，用于查找 Actor
        """
        super().__init__()
        self.config = config
        rollout_config = config.get("rollout", {})
        self._tp_size = rollout_config.get("tensor_model_parallel_size", 1)
        self._dp_size = dp_size
        self._dp_rank = dp_rank
        self.wg_prefix = wg_prefix

    async def init_engine(self):
        all_actors = ray.util.list_named_actors(all_namespaces=True)
        matched_actors = [actor for actor in all_actors if actor.get("name", None).startswith(self.wg_prefix + "WorkerDict_")]

        # TODO support multi node
        for matched_actor in matched_actors:
            current_rank = int(matched_actor["name"].split(":")[-1])
            print(f"chat_completion: self._dp_rank: {self._dp_rank}, self._tp_size: {self._tp_size}, current_rank: {current_rank}")

            # first_rank_in_node
            if current_rank == self._dp_rank * self._tp_size:
                self.worker = ray.get_actor(**matched_actor)
                break

        print(f"init_engine: self._dp_rank: {self._dp_rank}, self._tp_size: {self._tp_size}, self.worker: {self.worker}")

    async def chat_completion(self, raw_request: Request):
        request = await raw_request.json()
        print(f"chat_completion: raw_request.json(): {request}")
        output_future = self.worker.execute_method.remote("chat_completion", request)
        output = await output_future
        return JSONResponse(output)

    async def wake_up(self):
        pass

    async def sleep(self):
        pass
