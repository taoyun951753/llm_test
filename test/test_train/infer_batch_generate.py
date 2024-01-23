import os,sys
os.environ["CUDA_VISIBLE_DEVICES"]='0'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
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
import torch
from datasets import load_dataset, concatenate_datasets
import os,sys
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
import gc


# Setup logging
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO,  # if training_args.local_rank in [-1, 0] else logging.WARN,
                        handlers=[logging.StreamHandler(sys.stdout)], )
logger = logging.getLogger(__name__)

torch.cuda.empty_cache()
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

def hit_rate(outputs,labels):
    return len(set(outputs.split(';')).intersection(set(labels.split(";")))) / len(labels.split(";"))


def compute_hit(outputs: List[str], inputs: List[str], labels: List[str]):
    response = [s[len(sub):-1] for s, sub in zip(outputs, inputs) if len(s) > len(sub)]
    res = [hit_rate(s,l) for s, l in  zip(response,labels)]
    return res,len(res)
    
    

path = Path("/data/dataset/nlp-xingyu-dataset/recommed/test_data/")
files =["part-00001-3aa91bdc-f86c-4199-a63d-e5966fdef35a-c000.csv"]

for idx,file in enumerate(files):
    data_file = os.path.join(path, file)
    filename = ''.join(file.split(".")[:-1])
    file_type = 'csv'
    raw_dataset = load_dataset(file_type, data_files=data_file,delimiter="\t",
                               on_bad_lines='skip',column_names=['userId','feature','target'],
                                # cache_dir=cache_dir, 
                                keep_in_memory=False)
    
    llm_datasets = raw_dataset['train']


llm_datasets = llm_datasets.map(lambda examples: {'content': [x.replace('\\n','\n') for x in examples['feature']]},
                                                  batched=True,num_proc=16)

device = "cuda"

model_path = '/data/output_model/multi_data_and_qwen_7b_pt_4096/checkpoint-8750'
load_type=torch.float16
# load_type=torch.bfloat16
load_in_4bit=None
load_in_8bit=None

torch.cuda.empty_cache()

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
            # quantization_config=BitsAndBytesConfig(
            #     load_in_4bit=load_in_4bit,
            #     load_in_8bit=load_in_8bit,
            #     bnb_4bit_compute_dtype=load_type
            # )
            )#.to(device)
# 环境不对
# base_model.to_bettertransformer()

base_model = base_model.eval()


generation_config = GenerationConfig(
        temperature=0.2,
        top_k=40,
        top_p=0.9,
        do_sample=True,
        num_beams=1,
        repetition_penalty=1.2,
        max_new_tokens=512
    )


hit_res,hit_cnt = [],0
batch = 1
with torch.no_grad():
    for i in tqdm(range(3,20)):
        input_text,label_text = llm_datasets['content'][i * batch :(i+1) * batch],llm_datasets['target'][i * batch :(i+1) * batch]
        inputs = tokenizer.batch_encode_plus(input_text,return_tensors="pt",padding="longest",max_length=2000,truncation=True).to(device)
        # print(inputs)
        res = base_model.generate(
            input_ids = inputs["input_ids"].to(device),
            attention_mask = inputs['attention_mask'].to(device),
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            generation_config = generation_config
        )
        # print(res)
        outputs = tokenizer.batch_decode(res,skip_special_tokens=True)
        # print(outputs)
        hit_f,hit_c = compute_hit(outputs,input_text,label_text)
        hit_res.extend(hit_f)
        hit_cnt += hit_c
        logger.info(f"size of hit_res; {len(hit_res)}")
        logger.info(f"{i} step metric value: {sum(hit_res)/hit_cnt}")
        torch.cuda.empty_cache()
        gc.collect()
        
logger.info(f"final metric value: {sum(hit_res)/hit_cnt}")