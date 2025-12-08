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
"""
The main entry point to run the PPO algorithm
"""

import asyncio
import contextlib
import functools
import logging
import os

import torch
import torch.distributed
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp._unshard_param_utils import _get_module_fsdp_state, _unshard_params_for_summon
from torch.distributed.fsdp.api import FullStateDictConfig, ShardedStateDictConfig, StateDictType
from torch.distributed.fsdp.wrap import (
    _module_wrap_policy,
    transformer_auto_wrap_policy,
)
from transformers.trainer_pt_utils import get_module_class_from_name

from recipe.vla.models.vla_models import get_vla_model_and_config, get_vla_processor
from verl import DataProto
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.device import get_device_id, get_device_name, get_torch_device, set_expandable_segments
from verl.utils.flops_counter import FlopsCounter
from verl.utils.fsdp_utils import fsdp_version, init_fn
from verl.utils.import_utils import import_external_libs
from verl.utils.memory_utils import aggressive_empty_cache
from verl.utils.profiler import DistProfiler, log_gpu_memory_usage, simple_timer
from verl.utils.profiler.performance import reduce_timing, topk_reduce_ratio_min_max
from verl.workers.config.optimizer import build_optimizer
from verl.workers.fsdp_workers import ActorRolloutRefWorker, get_sharding_strategy

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

device_name = get_device_name()


def get_vla_fsdp_wrap_policy(module):
    """
    FSDP wrap policy for VLA models.

    Args:
        module: The model to wrap

    Returns:
        FSDP auto wrap policy function
    """

    # Get transformer layer classes to wrap
    if hasattr(module, "language_model"):
        # For VLA models, get transformer classes from language_model submodule
        fsdp_transformer_layer_cls_to_wrap = getattr(module.language_model, "_no_split_modules", None)
    else:
        # For standard models, get transformer classes directly from module
        fsdp_transformer_layer_cls_to_wrap = getattr(module, "_no_split_modules", None)

    # Build policies list
    policies = []

    # Add vision transformer policies for VLA models
    try:
        from timm.models.vision_transformer import VisionTransformer

        # Vision transformer policies
        vit_wrap_policy = functools.partial(_module_wrap_policy, module_classes={VisionTransformer})
        policies.append(vit_wrap_policy)
    except ImportError:
        pass

    # Prismatic projector policy for VLA models
    # The prismatic package initializes a DistributedOverwatch by default,
    # which initializes accelerate.PartialState, which in turn
    # initializes a torch.distributed process group in gloo.
    # This results in default group being gloo, which does not support CUDA tensors and allreduce average.
    # To fix this, we set the default group to cuda.
    try:
        from prismatic.extern.hf.modeling_prismatic import PrismaticProjector

        prismatic_fsdp_wrapping_policy = functools.partial(
            _module_wrap_policy,
            module_classes={PrismaticProjector},
        )
        policies.append(prismatic_fsdp_wrapping_policy)
    except ImportError:
        pass

    # Add transformer layer policies
    if fsdp_transformer_layer_cls_to_wrap is not None:
        transformer_cls_to_wrap = set()
        for layer_class in fsdp_transformer_layer_cls_to_wrap:
            transformer_cls = get_module_class_from_name(module, layer_class)
            if transformer_cls is None:
                raise Exception("Could not find the transformer layer class to wrap in the model.")
            else:
                transformer_cls_to_wrap.add(transformer_cls)

        llm_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            # Transformer layer class to wrap
            transformer_layer_cls=transformer_cls_to_wrap,
        )
        policies.append(llm_wrap_policy)

    # Return appropriate policy based on number of policies
    if len(policies) == 0:
        return None
    elif len(policies) == 1:
        return policies[0]
    else:
        # Multiple policies - combine with _or_policy
        from torch.distributed.fsdp.wrap import _or_policy

        return functools.partial(_or_policy, policies=policies)


class RobActorRolloutRefWorker(ActorRolloutRefWorker):
    """
    This worker can be instantiated as a standalone actor or a standalone rollout or a standalone reference policy
    or a hybrid engine based on the config.rollout
    """

    fsdp_unshard_exit_stack = contextlib.ExitStack()

    def _build_rollout(self, trust_remote_code=False):
        from recipe.vla.naive_rollout_rob import NaiveRolloutRob

        self.base_sync_done = False
        world_size = torch.distributed.get_world_size()
        dp = world_size
        infer_tp = self.config.rollout.tensor_model_parallel_size
        rollout_device_mesh = init_device_mesh(
            device_name, mesh_shape=(dp, infer_tp), mesh_dim_names=["dp", "infer_tp"]
        )
        # 3. init trainer and rollout random states
        self.torch_random_states = get_torch_device().get_rng_state()
        gen_dp_rank = rollout_device_mesh["dp"].get_local_rank()
        get_torch_device().manual_seed(gen_dp_rank + 1000)  # make sure all tp ranks have the same random states
        self.gen_random_states = get_torch_device().get_rng_state()
        get_torch_device().set_rng_state(self.torch_random_states)

        if torch.distributed.get_world_size() == 1 and fsdp_version(self.actor_module_fsdp) == 1:
            FSDP.set_state_dict_type(
                self.actor_module_fsdp,
                state_dict_type=StateDictType.FULL_STATE_DICT,
                state_dict_config=FullStateDictConfig(),
            )
        elif fsdp_version(self.actor_module_fsdp) == 1:
            FSDP.set_state_dict_type(
                self.actor_module_fsdp,
                state_dict_type=StateDictType.SHARDED_STATE_DICT,
                state_dict_config=ShardedStateDictConfig(),
            )
        else:
            raise NotImplementedError(f"Unsupported fsdp version {fsdp_version(self.actor_module_fsdp)}")

        self._register_dispatch_collect_info("rollout", dp_rank=self.rank, is_collect=True)
        self.rollout = NaiveRolloutRob(module=self.actor_module_fsdp, model_config=self.config.model)

        self.model_config = self.actor_model_config

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def switch_to_rollout(self):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.rollout_mode())
        log_gpu_memory_usage("After switch to rollout mode", logger=logger)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def switch_to_train(self):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.trainer_mode())
        log_gpu_memory_usage("After switch to trainer mode", logger=logger)

    async def rollout_mode(self):
        """Context switch hybridengine to rollout mode."""
        aggressive_empty_cache(force_sync=True)
        fsdp_unshard_exit_stack = contextlib.ExitStack()
        optional_state = _get_module_fsdp_state(self.actor_module_fsdp)
        if optional_state is None:
            self.fsdp_unshard_exit_stack = fsdp_unshard_exit_stack
        states_and_modules = ([optional_state], [self.actor_module_fsdp])

        self.base_sync_done = True
        # important: need to manually set the random states of each tp to be identical.
        self.torch_random_states = get_torch_device().get_rng_state()
        get_torch_device().set_rng_state(self.gen_random_states)
        for state, fsdp_module in zip(*states_and_modules, strict=False):
            fsdp_unshard_exit_stack.enter_context(
                _unshard_params_for_summon(
                    module=fsdp_module,
                    state=state,
                    writeback=False,
                    rank0_only=False,
                    offload_to_cpu=False,
                    with_grads=False,
                )
            )

        self.fsdp_unshard_exit_stack = fsdp_unshard_exit_stack
        logger.info("rollout mode")

    async def trainer_mode(self):
        """Context switch hybridengine to trainer mode."""

        self.actor_module_fsdp.train()

        # add empty cache after each compute
        aggressive_empty_cache(force_sync=True)

        set_expandable_segments(True)

        # restore random states
        self.gen_random_states = get_torch_device().get_rng_state()
        get_torch_device().set_rng_state(self.torch_random_states)
        if self.fsdp_unshard_exit_stack is not None:
            self.fsdp_unshard_exit_stack.close()
            self.fsdp_unshard_exit_stack = None
        logger.info("trainer mode")

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="rollout"), blocking=False)
    @DistProfiler.annotate(color="red", role="rollout_generate")
    def generate_sequences(self, prompts: DataProto):
        # Support all hardwares
        assert self._is_rollout
        prompts = prompts.to(get_device_id())

        meta_info = {
            "eos_token_id": self.model_config.generation_config.eos_token_id
            if self.model_config.generation_config is not None
            else self.model_config.tokenizer.eos_token_id,
            "pad_token_id": self.model_config.generation_config.pad_token_id
            if self.model_config.generation_config is not None
            else self.model_config.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)

        timing_generate = {}

        with simple_timer("generate_sequences", timing_generate):
            output = self.rollout.generate_sequences(prompts=prompts)

        timing_generate_topk_ratio, timing_generate_min, timing_generate_max = topk_reduce_ratio_min_max(
            timing_generate["generate_sequences"]
        )
        timing_generate = reduce_timing(timing_generate)
        timing_generate.update(
            {
                "generation_timing/max": timing_generate_max,
                "generation_timing/min": timing_generate_min,
                "generation_timing/topk_ratio": timing_generate_topk_ratio,
            }
        )
        output.meta_info["metrics"] = timing_generate
        output = output.to("cpu")

        # clear kv cache
        get_torch_device().empty_cache()
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        from recipe.vla.dp_rob import RobDataParallelPPOActor

        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))

        from omegaconf import OmegaConf

        override_model_config = OmegaConf.to_container(self.config.model.get("override_config", OmegaConf.create()))
        self.processor = get_vla_processor(self.config.model.path)

        if self._is_actor or self._is_rollout:
            # we need the model for actor and rollout
            if self._is_actor:
                optim_config = self.config.actor.optim
                fsdp_config = self.config.actor.fsdp_config
            else:
                optim_config = None
                fsdp_config = OmegaConf.create()
            self.actor_module_fsdp, self.actor_optimizer, self.actor_lr_scheduler, self.actor_model_config = (
                self._build_model_optimizer(
                    model_path=self.config.model.path,
                    fsdp_config=fsdp_config,
                    optim_config=optim_config,
                    override_model_config=override_model_config,
                    enable_gradient_checkpointing=self.config.model.get("enable_gradient_checkpointing", False),
                    trust_remote_code=self.config.model.get("trust_remote_code", False),
                )
            )

            if fsdp_version(self.actor_module_fsdp) == 1:
                # get the original unwrapped module
                self.actor_module = self.actor_module_fsdp._fsdp_wrapped_module

        if self._is_actor:
            OmegaConf.set_struct(self.config.actor, True)
            self.actor = RobDataParallelPPOActor(
                config=self.config.actor, actor_module=self.actor_module_fsdp, actor_optimizer=self.actor_optimizer
            )

        if self._is_rollout:
            self._build_rollout(trust_remote_code=self.config.model.get("trust_remote_code", False))

        if self._is_actor:
            self.flops_counter = FlopsCounter(self.actor_model_config)
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.actor_module_fsdp,
                optimizer=self.actor.actor_optimizer,
                lr_scheduler=self.actor_lr_scheduler,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
                checkpoint_config=self.config.actor.checkpoint,
            )

        torch.distributed.barrier()

    def _build_model_optimizer(
        self,
        model_path,
        fsdp_config,
        optim_config,
        override_model_config,
        use_remove_padding=False,
        use_fused_kernels=False,
        enable_gradient_checkpointing=False,
        trust_remote_code=False,
        use_liger=False,
        role="actor",
        enable_activation_offload=False,
    ):
        from torch.distributed.fsdp import CPUOffload, MixedPrecision

        from verl.utils.model import print_model_size
        from verl.utils.torch_dtypes import PrecisionType

        assert role in ["actor", "ref"]

        log_gpu_memory_usage(f"Before init {role} from HF AutoModel", logger=logger)
        local_path = model_path

        # # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        # # TODO(zhangchi.usc1992): 1. support create from random initialized model. 2. Support init with FSDP directly
        # self.tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        # self.processor = hf_processor(local_path, trust_remote_code=trust_remote_code)

        torch_dtype = fsdp_config.get("model_dtype", None)
        if torch_dtype is None:
            torch_dtype = torch.float32 if self._is_actor else torch.bfloat16
        else:
            torch_dtype = PrecisionType.to_dtype(torch_dtype)

        actor_module, actor_model_config = get_vla_model_and_config(
            local_path, trust_remote_code, override_model_config
        )

        # some parameters may not in torch_dtype. TODO(zhangchi.usc1992) remove this after we switch to fsdp2
        actor_module.to(torch_dtype)

        torch.distributed.barrier()

        if self.rank == 0:
            print_model_size(actor_module)

        log_gpu_memory_usage(f"After init {role} from HF AutoModel", logger=logger)

        # We wrap FSDP for rollout as well
        mixed_precision_config = fsdp_config.get("mixed_precision", None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get("param_dtype", "bf16"))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get("reduce_dtype", "fp32"))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get("buffer_dtype", "fp32"))
        else:
            param_dtype = PrecisionType.to_dtype(fsdp_config.dtype)
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

        auto_wrap_policy = get_vla_fsdp_wrap_policy(module=actor_module)

        if self.rank == 0:
            print(f"wrap_policy: {auto_wrap_policy}")

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        # TODO: add transformer policy
        # We force reference policy to use CPUOffload to save memory.
        # We force turn off CPUOffload for actor because it causes incorrect results when using grad accumulation
        cpu_offload = None if role == "actor" else CPUOffload(offload_params=True)
        fsdp_strategy = self.config.actor.strategy
        if fsdp_strategy == "fsdp":
            actor_module_fsdp = FSDP(
                actor_module,
                cpu_offload=cpu_offload,
                param_init_fn=init_fn,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_device_id(),
                sharding_strategy=sharding_strategy,  # zero3
                mixed_precision=mixed_precision,
                sync_module_states=True,
                device_mesh=self.device_mesh,
                use_orig_params=self.use_orig_params,
                forward_prefetch=fsdp_config.get("forward_prefetch", False),
            )
        else:
            raise NotImplementedError(f"not implement {fsdp_strategy}")

        # if enable_activation_offload:
        #     enable_activation_offloading(actor_module_fsdp, fsdp_strategy, enable_gradient_checkpointing)

        log_gpu_memory_usage(f"After {role} FSDP init", logger=logger)

        # TODO: add more optimizer args into config
        if role == "actor" and optim_config is not None:
            from verl.utils.torch_functional import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

            actor_optimizer = build_optimizer(actor_module_fsdp.parameters(), optim_config)

            total_steps = optim_config.get("total_training_steps", 0)
            num_warmup_steps = int(optim_config.get("lr_warmup_steps", -1))
            lr_scheduler_type = optim_config.get("lr_scheduler_type", "constant")
            min_lr_ratio = optim_config.get("min_lr_ratio", 0.0)
            num_cycles = optim_config.get("num_cycles", 0.5)
            if num_warmup_steps < 0:
                num_warmup_steps_ratio = optim_config.get("lr_warmup_steps_ratio", 0.0)
                num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

            if self.rank == 0:
                print(f"Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}")

            if lr_scheduler_type == "constant":
                actor_lr_scheduler = get_constant_schedule_with_warmup(
                    optimizer=actor_optimizer, num_warmup_steps=num_warmup_steps
                )
            elif lr_scheduler_type == "cosine":
                actor_lr_scheduler = get_cosine_schedule_with_warmup(
                    optimizer=actor_optimizer,
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=total_steps,
                    min_lr_ratio=min_lr_ratio,
                    num_cycles=num_cycles,
                )
            else:
                raise NotImplementedError(f"LR scheduler type {lr_scheduler_type} is not supported")

            log_gpu_memory_usage(f"After {role} optimizer init", logger=logger)
        else:
            actor_optimizer = None
            actor_lr_scheduler = None

        return actor_module_fsdp, actor_optimizer, actor_lr_scheduler, actor_model_config
