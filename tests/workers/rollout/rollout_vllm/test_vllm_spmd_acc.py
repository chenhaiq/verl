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


import torch
from vllm import LLM, SamplingParams

from tests.workers.rollout.utils_sglang import (
    are_lists_similar,
    clean_torchelastic_env,
    generate_hf_output,
    initialize_global_process_group,
    load_tokenizer_and_model,
    prepare_inputs,
)


def test_vllm_spmd():
    assert torch.cuda.device_count() >= 2, "At least 2 GPUs is required to run tp+dp tests."
    initialize_global_process_group(spmd=True)
    clean_torchelastic_env()
    # Initialize model and token
    local_model_path = "/file_system/common-models/Qwen/Qwen3-8B"
    tokenizer, actor_model = load_tokenizer_and_model(local_model_path)

    # fill rollout config
    max_prompt_length = 1024
    max_response_length = 128

    preencode_prompts = [
        """<|im_start|>system

The assistant is designed to be helpful, proactive, and highly interactive.


The assistant strives to accurately interpret the user's intent throughout the conversation, acknowledging previous interactions to maintain context and continuity. If the user's message is unclear or lacks necessary details, the assistant always asks for clarification rather than making assumptions. For example, if the user's request is incomplete, the assistant responds with: "Could you provide more details so I can assist you better?"


The assistant asks specific follow-up questions and offers suggestions based on the user's needs, avoiding vague or generic prompts. It proactively provides guidance and potential next steps, especially in complex tasks such as writing, analysis, coding, and question answering.


The assistant is mindful of how much content the user needs to read or type, keeping interactions concise and efficient. It reduces unnecessary repetition and ensures responses are relevant, well-structured, and free from errors. When presenting options or asking for feedback, the assistant simplifies interactions by offering multiple-choice answers or specific suggestions to make it easier for the user to respond quickly.


The assistant adapts its tone to align with the user's emotional state and style, adjusting its approach as needed. If uncertain about something, the assistant honestly says, "I don't know," and suggests ways for the user to find the information.


The assistant provides factually accurate, coherent, and relevant responses, using proper grammar and structure. It remains interactive and proactive across all tasks, continually seeking feedback to refine and improve interactions.<|im_end|>

<|im_start|>user

Hey, I'm trying to figure out which of these functions are invertible. Can you help me understand how to determine this from the graphs?<|im_end|>

<|im_start|>assistant

To determine if a function is invertible from its graph, you can use the Horizontal Line Test. A function is invertible if and only if no horizontal line intersects the graph of the function more than once. This means the function should be one-to-one (bijective). If a horizontal line intersects the function's graph at more than one point, the function is not invertible.


Do you have specific graphs or functions you'd like to examine for invertibility? If you can share more details or describe the graphs, I can help you analyze them further. Also, if you'd like an example of a function and its graph to better understand this concept, let me know!<|im_end|>

<|im_start|>user

Okay, thanks for the explanation. I'm still a bit confused about applying the Horizontal Line Test to these specific graphs. Can you help me check them one by one to see which ones are invertible? I'm not quite sure how to tell just by looking.<|im_end|>

<|im_start|>assistant"""
    ]

    input_ids, attention_mask, _ = prepare_inputs(tokenizer, preencode_prompts, max_prompt_length)
    hf_response_tokens = generate_hf_output(actor_model, input_ids, attention_mask, tokenizer, max_response_length)
    print(f"hf response: {hf_response_tokens}")

    input_ids = input_ids.cuda()
    attention_mask = attention_mask.cuda()

    kwargs = dict(
        n=1,
        logprobs=0,
        repetition_penalty=1.0,
        max_tokens=max_response_length,
        temperature=0,
    )

    tensor_parallel_size = 4

    sampling_params = SamplingParams(**kwargs)
    llm = LLM(
        model=local_model_path,
        enable_sleep_mode=True,
        tensor_parallel_size=tensor_parallel_size,
        distributed_executor_backend="external_launcher",
        dtype="bfloat16",
        enforce_eager=True,
        gpu_memory_utilization=0.8,
        disable_custom_all_reduce=True,
        skip_tokenizer_init=False,
        enable_prefix_caching=True,
        trust_remote_code=True,
        seed=1,
        enable_chunked_prefill=True,
    )

    outputs = llm.generate(preencode_prompts, sampling_params=sampling_params, use_tqdm=False)
    vllm_response_tokens = []
    for output in outputs:
        generated_text = output.outputs[0].text
        vllm_response_tokens.append(generated_text)

    if torch.distributed.get_rank() == 0:
        print(f"vllm response: {vllm_response_tokens}")

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()
    assert are_lists_similar(hf_response_tokens, vllm_response_tokens, threshold=10), "Strings differ more than 10%:\n"
    print("Check Pass")


if __name__ == "__main__":
    test_vllm_spmd()
