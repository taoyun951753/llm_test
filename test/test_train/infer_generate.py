import os,sys
os.environ["CUDA_VISIBLE_DEVICES"] = '1'  
import torch

from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    LlamaForCausalLM,
    LlamaTokenizer,
    LlamaConfig,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    is_torch_tpu_available,
    set_seed,
    BitsAndBytesConfig,
    GenerationConfig
)
from pathlib import Path
import datasets

from datasets import load_dataset, concatenate_datasets

import glob
from tqdm import tqdm
from accelerate import init_empty_weights,infer_auto_device_map,load_checkpoint_in_model,dispatch_model
import gc

from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import send_example_telemetry
from transformers.utils.versions import require_version

from sklearn.metrics import accuracy_score
from peft import LoraConfig, TaskType, get_peft_model, PeftModel, get_peft_model_state_dict
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

import logging
import numpy as np
import math

from dataclasses import dataclass, field
from itertools import chain
from typing import Optional, List, Dict, Any, Mapping

path = "/data/dataset-sft/recommend/feature2item.json"
raw_dataset = load_dataset("json", data_files=path,
                                keep_in_memory=False)

llm_data = raw_dataset['train']

model_path = '/data/jesse/checkpoint-2000-tmp/'
load_type=torch.float16

tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=True, trust_remote_code=True)
tokenizer.padding_side = "left"
# 适配 Qwen 模型
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eod_id
if tokenizer.eos_token_id is None:
    tokenizer.eos_token_id = tokenizer.eod_id

base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=load_type,
            low_cpu_mem_usage=True,
            device_map='auto',
            trust_remote_code=True
            )

if torch.cuda.is_available():
    # device = torch.device(0)
    device = "cuda"
else:
    device = torch.device('cpu')

generation_config = GenerationConfig(
        temperature=0.5,
        top_p=0.9,
        do_sample=True,
        num_beams=1,
        repetition_penalty=1.2,
        max_new_tokens=1024,
        max_length=4096,
        no_repeat_ngram_size=6,
        num_return_sequences=3
    )


with torch.no_grad():
    # for i in tqdm(range(7)):
    # for i in tqdm(range(9,12)):
    # input_text,label_text = llm_datasets['content'][i * batch :(i+1) * batch],llm_datasets['target'][i * batch :(i+1) * batch]
    input_text,label_text = llm_data['instruction'][0],llm_data['output'][0]

    inputs = tokenizer.encode(input_text,return_tensors="pt").to(device)
    # print(inputs.shape)  # torch.Size([1, 365])
    # print(inputs)
    res = base_model.generate(
        inputs,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        generation_config = generation_config
    )
    # print(res.shape)   # torch.Size([1, 366])
    outputs = tokenizer.decode(res[0],skip_special_tokens=True)
    print(outputs)