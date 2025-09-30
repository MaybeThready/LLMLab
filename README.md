# Efficient Instruction Tuning of GPT-2: From Model Construction to Evaluation of LoRA Methods
## 项目概述
本项目是论文 Efficient Instruction Tuning of GPT-2: From Model Construction to Evaluation of LoRA Methods 的附属项目，包含了论文实验的复现代码。所有代码均可在消费级GPU（RTX4060）上运行。
 
## 环境依赖

Python 版本为 3.12.7.

第三方库详见 requirements.txt. 其中 Pytorch 环境建议手动配置，其他第三方库均可通过 pip 安装。
 
## 目录结构
```
LLMLab
├── README.md
├── data                            # 数据文件
│   ├── datasets                    # 数据集
│   │   ├── instruct
│   │   │   └── alpaca_data.json
│   └── models                      # 模型文件
│       ├── instruct                # 微调模型文件
│       │   ├── v1
│       │   │   └── log
│       │   ├── v2
│       │   │   └── log
│       │   ├── v3
│       │   │   └── log
│       │   └── v4
│       │       └── log
│       ├── pretrained              # 预训练权重
│       │   ├── gpt2-124M.pth
│       │   ├── gpt2-1558M.pth
│       │   ├── gpt2-355M.pth
│       │   └── gpt2-774M.pth
├── requirements.txt
└── src                             # 源代码文件
    ├── Models                      # 模型库
    │   ├── __init__.py
    │   ├── config.py               # 配置模块
    │   ├── dataset                 # 数据集模块
    │   │   ├── __init__.py
    │   │   └── instruct.py         # 偏好微调数据处理
    │   ├── load_weight.py          # 加载tensorflow预训练文件
    │   ├── models
    │   │   ├── __init__.py
    │   │   ├── instruct.py         # 指令微调核心代码
    │   │   ├── pretrained.py       # 预训练模型核心代码
    │   │   └── tokenizer.py        # 分词器加载代码
    │   ├── networks                # 基础网络架构
    │   │   ├── __init__.py
    │   │   ├── gpt.py
    │   │   ├── lora.py
    │   │   └── qwen.py
    │   └── utils.py
    ├── config.py                   # 配置模块
    ├── instruct_tuning.py          # 指令微调入口
    └── main.py                     # 主程序入口
```


## 使用说明
本项目采用 Python 包架构进行开发，读者可以根据自己的实际需求调用相关函数。

### 1. 下载并转换 GPT-2 模型权重
```python
from Models.load_weight import download_gpt2, transform_gpt2_params_to_torch
from config import GPT2_124M_CONFIG

MODEL_SIZE = "124M"  # 支持 124M, 355M, 774M, 1558M 四种规格
TF_DST_DIR = "tensorflow_dst_model_path"
PT_DST_PATH = "pytorch_dst_model_path.pth"

download_gpt2(MODEL_SIZE, TF_DST_DIR)
transform_gpt2_params_to_torch(TF_DST_DIR, PT_DST_PATH, GPT2_124M_CONFIG)
```

### 2. 与预训练模型进行对话
运行 src/main.py 即可。其中，修改
```python
GPT2_CONFIG = GPT2_124M_CONFIG
```
可以更改使用的参数规模，修改
```python
GPT2_MODEL_PATH = abspath("../data/models/pretrained/gpt2-124M.pth")
```
可以更改预训练模型权重位置。

在程序入口处可以选择调用的函数，包括预训练文本生成和预训练用户对话功能。

### 3. 指令微调
src/instruct_tuning.py 提供了我们实验的指令微调示例程序，用户可以直接运行，也可以自己编写个性化程序进行训练。

### 4. 评估结果
用户可以用 tensorboard 可视化训练数据。示例程序中，训练数据保存在 data/models/instruct/v2/log 中。

用户也可以自己仿照示例程序自己编写代码，与微调后的模型进行对话。我们鼓励用户在我们开发的 Python 包的基础上进一步构建代码，实现更复杂的功能。
