# LLMLab：To Create a Chatbot
## 1 项目概述
### 1.1 项目性质
LLMLab是一个用于应付《人工智能导论》课程期末论文的实验项目。
采用Pytorch开发，旨在通过加载LLM预训练模型进行微调， 
开发一个类似于Chatbot的应用程序。也可以更近一步，开发机器客服、数字分身等应用。

### 1.2 项目结构
截至8月14日，LLMLab的项目结构大致如下：
```
LLMLab
| data                          # 数据文件夹
    | models                    # 模型文件夹
        | pretrained            # 预训练模型
            | openai            # openai类预训练模型
            | qwen              # qwen类预训练模型
            gpt2-124M.pth       # gpt2-124M pytorch参数
            qwen2p5-1p5B.pth    # qwen2p5-1p5B pytorch参数
        | tokenizer             # 分词器json文件
            Chinese-LLaMA.json
            DeepSeek.json
            T5_Pegasus.json
            qwen2p5-1p5B.json
| src                           # 源代码文件夹
    | Applications              # 应用包
    | Models                    # 模型包
        | models                # 模型模块
            __init__.py
            pretrained.py       # 预训练模型
            tokenizer.py        # 分词器模型
        | networks              # 网络模块
            __init__.py
            gpt.py
            qwen.py
        config.py               # 模型/网络配置接口
        load_weight.py          # 权重加载模块
        utils.py                # 实用工具
    config.py                   # 模型/网络配置
    main.py                     # 主程序入口
    test.py                     # 测试程序
    utils.py                    # 实用工具
.gitignore
README.md
requirements.txt
```
下面对各个部分进行详细解释。

#### 1.2.1 数据文件
数据文件包括模型参数、数据集等。通常来说，这部分文件不会上传到github上。我们可以通过QQ群进行传输。

预计后续会在data目录下创建一个与models文件夹同级的文件夹dataset，用于存储数据集。

预计后续会在models目录下创建一个与pretrained文件夹同级的文件夹fine-tuning用于存储微调数据。

以此类推，只要在新建文件夹时符合命名规范、结构逻辑、格式统一就行。

#### 1.2.2 Applications
应用包是对模型应用的开发，比如客服UI搭建、群聊接口、MCP等。该包的内部结构待定。

#### 1.2.3 Models
模型包负责模型的搭建与训练、数据预处理等。

子包models负责以类的形式开发各种模型，比如gpt模型、tokenizer模型等。

子包networks定义的是具体的神经网络。

可能会增加子包dataset，负责数据集处理。

模块utils是在开发Models包时可能会用到的一些实用函数。

模块config负责以类的形式管理模型的各种参数。该类会在外层config文件中得到实例化。

模块load_weight负责处理预训练模型权重。

#### 1.2.4 程序入口
config模块负责实例化模型的config类。

main是程序的入口，在该文件中调用Models包或Applications包的函数/类以完成功能。

test是测试程序入口，自用，无需上传到github上。

utils是主程序会用的的一些实用函数。

## 2 待完成的功能

### 2.1 更多的模型权重
可以尝试加载更多模型的权重，以更好地支持中文聊天。

### 2.2 指令微调数据集
目前尚未寻找到较好的指令微调数据集，也尚未进行数据预处理。

### 2.3 偏好微调数据集构建
目前尚未完成用其他LLM构建偏好微调数据集的代码。

### 2.4 指令微调
尚未构建LoRA微调的代码，也没有进行任何训练。

### 2.5 偏好微调
尚未构建偏好微调的代码，也没有进行任何训练。

### 2.6 Agent构建
尚未将模型包装成一个智能体从而进行交互。

### 2.7 上下文记忆与查询
尚未为模型添加长短期记忆功能。

### 2.8 知识库（可选）
建议为模型添加知识库以解决部分专业性较强的问题。

### 2.9 MCP（可选）
为模型提供强大的MCP使它更厉害！

### 2.10 调用接口（可选）
目前所有的对话都是发生在终端的，是否可以考虑将机器人接入QQ等平台，或者自己搭建一个平台用于聊天？