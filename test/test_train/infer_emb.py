import pandas as pd 
import numpy as np 
# import faiss
import os,sys
os.environ["CUDA_VISIBLE_DEVICES"] = '0'  
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
import torch
import torch.nn.functional as F

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

llm_datasets = [
    "怎样培养孩子爱学习？",
    "给我推荐5本悬疑专辑",
    "适合给宝宝睡觉时听的音乐",
    "指数基金学习",
    "脂肪如何燃烧",
    "推荐几本穿越小说",
    "推荐几本历史解读的专辑",
    "有什么好听的相声",
    "《胡椒焦点局》黑胡椒粉粉，热衷于八卦热点，喜欢瞎叭叭。节目亮点：粗枝烂叶的普通人视角叭叭热点。像朋友聊天一样输出一些又浅显又无知的观点>。不喜请喷！根据这些内容给我推荐一些相似内容",
    "《疗愈｜冥想》 所有的疗愈都是疗愈我们自己，我们需要通过冥想去认识到自己的本性，与真实的自己链接，冥想是最好的与本性链接的方式，冥想不仅仅能够让你更好的专注当下，冥想是认识真相的工具。 根据这些内容给我推荐一些相似内容",
    "《硬核狠人》苏联的小偷是什么格局 苏联狙击手最怕谁？谁是美国终结者战士？好莱坞明星怎么搞对象？根据这些内容给我推荐一些相似内容",
    "《女人劝你要善良100位真实女人自述爱情婚姻故事》我教你怎么应对家暴出轨, 根据这些简介给我推荐一些相似内容",
    "《财务职场分享》 企业的车出差违章被罚如何入账？3付款方承担汇票的贴息利息该如何入账呢？2给牛买了保险，牛没了，收到保险赔付如何入账？1新>准则下租赁提前终止后续如何入账？根据这些简介给我推荐一些相似内容"
]


model_path = "/data/output_model_sft/test-qwen-sft-yzy/checkpoint-15-half"
tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=True, padding_side='left',trust_remote_code=True)
load_type=torch.float16
device = "cuda"
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
            ).eval()
base_model.generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True)


with torch.no_grad():
    input_text= llm_datasets[:5]
    inputs = tokenizer(input_text,return_tensors="pt",padding="longest",max_length=4096,truncation=True).to(device)
    print(inputs['input_ids'].shape)
    res = base_model(**inputs,output_hidden_states=True)
    # print(res.hidden_states)
    hiddens = res.hidden_states[-1]
    totals = [hidden[mask==1].mean(dim=0,dtype=torch.float16) for mask,hidden in zip(inputs['attention_mask'],hiddens)]
    print(hiddens.shape)
    
    inputs2 = tokenizer(input_text[:2],return_tensors="pt",padding="longest",max_length=4096,truncation=True).to(device)
    res2 = base_model(**inputs2,output_hidden_states=True)
    hiddens2 = res2.hidden_states[-1]
    totals2 = [hidden[mask==1].mean(dim=0,dtype=torch.float16) for mask,hidden in zip(inputs2['attention_mask'],hiddens2)]
    
    
    
    singles = []
    singles_2 = []
    for input in input_text:
        # token_input = tokenizer.encode(input,return_tensors="pt").to(device)
        token_input = tokenizer([input],return_tensors="pt",padding="longest",max_length=4096,truncation=True).to(device)
        print(f"token_input shape : {token_input['input_ids'].shape}")
        single_res = base_model(token_input['input_ids'],output_hidden_states=True)
        
        token_input_2 = tokenizer.encode(input,return_tensors="pt").to(device)
        single_res_2 = base_model(token_input_2,output_hidden_states=True)
        singles.append(single_res.hidden_states[-1])
        singles_2.append(single_res_2.hidden_states[-1])
        
print("----")
[print(sum(abs(total - total2 < 0.0001))) for total,total2 in zip(totals[:2],totals2)]
print("----")
[print(sum(abs(total - single.squeeze(0).mean(dim=0,dtype=torch.float16) < 0.001))) for total,single in zip(totals2,singles_2[:2])]
print("----")
[print(sum(abs(single_2.squeeze(0).mean(dim=0,dtype=torch.float16) - single.squeeze(0).mean(dim=0,dtype=torch.float32) < 0.0001))) for single_2,single in zip(singles_2,singles)]