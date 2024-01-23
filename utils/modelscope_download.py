from modelscope.pipelines import pipeline
import torch
from modelscope import Model

# model = Model.from_pretrained('ZhipuAI/chatglm3-6b', revision='v1.0.0', device_map='auto', torch_dtype=torch.float16)

# model = Model.from_pretrained("qwen/Qwen-1_8B", revision='master', device_map="auto", trust_remote_code=True,torch_dtype=torch.float16)

model = Model.from_pretrained('/data/yun.tao/chinese-roberta-wwm-ext',revision='v1.0.0',device_map='auto', torch_dtype=torch.float16)
print(model)
