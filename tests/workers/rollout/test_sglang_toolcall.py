# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
usage: torchrun --standalone --nnodes=1 \
    --nproc_per_node=1 $(which pytest) \
    -s test_sglang_toolcall.py
"""

import asyncio
import json

import torch
from sglang.srt.entrypoints.engine import Engine
from utils_sglang import (
    clean_torchelastic_env,
    generate_hf_output,
    initialize_global_process_group,
    load_tokenizer_and_model,
)

from verl.tools.utils.tool_registry import initialize_tools_from_config


def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor):
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


def test_sglang_spmd():
    initialize_global_process_group(spmd=True)
    clean_torchelastic_env()

    max_response_length = 16

    local_model_path = "Qwen/Qwen2.5-1.5B-Instruct"

    tool_config = {
        "tools": [
            {
                "class_name": "tests.workers.rollout.rollout_vllm.test_vllm_chat_scheduler.WeatherTool",
                "config": {"type": "native"},
            },
            {
                "class_name": "tests.workers.rollout.rollout_vllm.test_vllm_chat_scheduler.WeatherToolWithData",
                "config": {"type": "native"},
            },
        ]
    }
    tool_config_path = "/tmp/tool_config.json"
    with open(tool_config_path, "w") as f:
        json.dump(tool_config, f)

    tool_list = initialize_tools_from_config(tool_config_path)
    tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]

    tokenizer, actor_model = load_tokenizer_and_model(local_model_path)

    messages = [
        {"role": "user", "content": "What's the temperature in Beijing now?"},
    ]
    input_ids = tokenizer.apply_chat_template(messages, tools=tool_schemas, add_generation_prompt=True, return_tensors="pt", padding=True)

    llm = Engine(
        model_path=local_model_path,
        dtype="bfloat16",
        mem_fraction_static=0.5,
        enable_memory_saver=True,
        tp_size=1,
    )

    _input_ids = input_ids.cuda()
    idx_list = []

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    for i in range(_input_ids.shape[0]):
        idx_list.append(_pre_process_inputs(pad_token_id, _input_ids[i]))

    sampling_params = dict(
        n=1,
        temperature=0,
        top_p=1,
        top_k=-1,
        max_new_tokens=max_response_length,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        repetition_penalty=1.0,
        skip_special_tokens=True,
        spaces_between_special_tokens=True,
        ignore_eos=False,
    )

    loop = asyncio.get_event_loop()
    [output] = loop.run_until_complete(llm.async_generate(input_ids=idx_list, sampling_params=sampling_params))

    sglang_response_text = output["text"]

    print(f"sglang response: {sglang_response_text}")
    attention_mask = torch.ones_like(input_ids)
    [hf_response_text] = generate_hf_output(actor_model, input_ids, attention_mask, tokenizer, max_response_length)
    print(f"hf response: {hf_response_text}")
    assert are_lists_similar(hf_response_text, sglang_response_text), "Strings differ more than 50%:\n"
    print("SPMD Test Passed!")

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


def are_lists_similar(a, b, threshold=50):
    total_length = 0
    total_diff = 0
    total_length = min(len(a), len(b))
    for i in range(total_length):
        if a[i] != b[i]:
            total_diff += 1
    percentage_difference = (total_diff / total_length) * 100
    print(f"Total difference: {percentage_difference:.2f}%")
    return percentage_difference <= threshold
