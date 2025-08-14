# -*- coding: UTF-8 -*-

# 将模型权重转换为pytorch权重数据

import os
import urllib.request

import json
import numpy as np
import tensorflow as tf
import torch
from safetensors import safe_open
from rich import print
from rich.rule import Rule
from rich.progress import Progress

from .config import GPTNetworkConfig, QwenNetworkConfig
from .networks import GPTNetwork, QwenNetwork
from .utils import assign


def download_and_load_gpt2(model_size: str, models_dir: str):
    """
    下载gpt2权重数据
    :param model_size: 支持124M、335M、774M、1558M四种模型权重
    :param models_dir: 模型下载位置
    :return:
    """
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size not in {allowed_sizes}")

    model_dir = os.path.join(models_dir, model_size)
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    backup_base_url = "https://f001.backblazeb2.com/file/LLMs-from-scratch/gpt2"
    filenames = [
        "checkpoint", "encoder.json", "hparams.json",
        "model.ckpt.data-00000-of-00001", "model.ckpt.index",
        "model.ckpt.meta", "vocab.bpe"
    ]

    os.makedirs(model_dir, exist_ok=True)
    print(Rule("Start Downloading"))
    for filename in filenames:
        file_url = os.path.join(base_url, model_size, filename)
        backup_url = os.path.join(backup_base_url, model_size, filename)
        file_path = os.path.join(model_dir, filename)
        download_file(file_url, file_path, backup_url)
    print(Rule("End Downloading"))


def download_file(url: str, destination: str, backup_url=None):
    """
    下载模型权重
    :param url:
    :param destination:
    :param backup_url:
    :return:
    """
    def _attempt_download(download_url):
        with urllib.request.urlopen(download_url) as response:
            file_size = int(response.headers.get("Content-Length", 0))

            if os.path.exists(destination):
                file_size_local = os.path.getsize(destination)
                if file_size == file_size_local:
                    print(f"File already exists and is up-to-date: {destination}")
                    return True

            block_size = 1024

            progress_bar_description = os.path.basename(download_url)
            with Progress() as progress:
                progress_bar = progress.add_task(description=progress_bar_description, total=file_size)
                with open(destination, "wb") as file:
                    while True:
                        chunk = response.read(block_size)
                        if not chunk:
                            break
                        file.write(chunk)
                        progress.advance(progress_bar, len(chunk))
            return True

    try:
        if _attempt_download(url):
            return
    except (urllib.error.HTTPError, urllib.error.URLError):
        if backup_url is not None:
            print(f"Primary URL ({url}) failed. Attempting backup URL: {backup_url}")
            try:
                if _attempt_download(backup_url):
                    return
            except urllib.error.HTTPError:
                pass

        error_message = (
            f"Failed to download from both primary URL ({url})"
            f"{' and backup URL (' + backup_url + ')' if backup_url else ''}."
            "\nCheck your internet connection or the file availability.\n"
        )
        print(error_message)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def load_gpt2_params_from_tf_ckpt(ckpt_path: str, settings: dict):
    """
    用tensorflow加载GPT权重
    :param ckpt_path:
    :param settings:
    :return:
    """
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    for name, _ in tf.train.list_variables(ckpt_path):
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))
        variable_name_parts = name.split("/")[1:]
        target_dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]

        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array

    return params


def transform_gpt2_params_to_torch(src: str, fdst: str, config: GPTNetworkConfig):
    """
    将tensorflow权重转化为pytorch权重
    :param src: tensorflow权重文件夹
    :param fdst: pytorch权重文件路径
    :param config:
    :return:
    """
    tf_ckpt_path = tf.train.latest_checkpoint(src)
    settings = json.load(open(os.path.join(src, "hparams.json"), "r", encoding="utf-8"))
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)
    gpt: GPTNetwork = GPTNetwork(config)

    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])

    for b in range(len(params["blocks"])):
        # 自注意力模块
        q_w, k_w, v_w = np.split((params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.transformers[b].attention.W_query.weight = assign(gpt.transformers[b].attention.W_query.weight, q_w.T)
        gpt.transformers[b].attention.W_key.weight = assign(gpt.transformers[b].attention.W_key.weight, k_w.T)
        gpt.transformers[b].attention.W_value.weight = assign(gpt.transformers[b].attention.W_value.weight, v_w.T)
        q_b, k_b, v_b = np.split((params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.transformers[b].attention.W_query.bias = assign(gpt.transformers[b].attention.W_query.bias, q_b)
        gpt.transformers[b].attention.W_key.bias = assign(gpt.transformers[b].attention.W_key.bias, k_b)
        gpt.transformers[b].attention.W_value.bias = assign(gpt.transformers[b].attention.W_value.bias, v_b)
        gpt.transformers[b].attention.out_proj.weight = assign(
            gpt.transformers[b].attention.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T
        )
        gpt.transformers[b].attention.out_proj.bias = assign(
            gpt.transformers[b].attention.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"]
        )

        # 前馈网络模块
        gpt.transformers[b].feedforward.layers[0].weight = assign(
            gpt.transformers[b].feedforward.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T
        )
        gpt.transformers[b].feedforward.layers[0].bias = assign(
            gpt.transformers[b].feedforward.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"]
        )
        gpt.transformers[b].feedforward.layers[2].weight = assign(
            gpt.transformers[b].feedforward.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T
        )
        gpt.transformers[b].feedforward.layers[2].bias = assign(
            gpt.transformers[b].feedforward.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"]
        )

        # 层归一化模块
        gpt.transformers[b].norm1.weight = assign(
            gpt.transformers[b].norm1.weight,
            params["blocks"][b]["ln_1"]["g"]
        )
        gpt.transformers[b].norm1.bias = assign(
            gpt.transformers[b].norm1.bias,
            params["blocks"][b]["ln_1"]["b"]
        )
        gpt.transformers[b].norm2.weight = assign(
            gpt.transformers[b].norm2.weight,
            params["blocks"][b]["ln_2"]["g"]
        )
        gpt.transformers[b].norm2.bias = assign(
            gpt.transformers[b].norm2.bias,
            params["blocks"][b]["ln_2"]["b"]
        )

    # 输出层
    gpt.final_norm.weight = assign(gpt.final_norm.weight, params["g"])
    gpt.final_norm.bias = assign(gpt.final_norm.bias, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])

    torch.save(gpt.state_dict(), fdst)
    print("Successfully transform weights!")


def transform_qwen2p5_params_to_torch(fsrc: str, fdst: str, config: QwenNetworkConfig):
    """
    将safetensors权重转化为pytorch权重
    :param fsrc: safetensors权重文件路径
    :param fdst: pytorch权重文件路径
    :param config:
    :return:
    """
    params = {}
    with safe_open(fsrc, framework="pt", device="cpu") as f:
        for k in f.keys():
            params[k] = f.get_tensor(k)
    qwen: QwenNetwork = QwenNetwork(config)
    qwen.tok_emb.weight = assign(qwen.tok_emb.weight, params["model.embed_tokens.weight"])
    for layer in range(config.num_layers):
        layer_name = f"model.layers.{layer}."
        qwen.transformers[layer].norm1.weight = assign(
            qwen.transformers[layer].norm1.weight, params[layer_name + "input_layernorm.weight"]
        )

        qwen.transformers[layer].feedforward.W_down.weight = assign(
            qwen.transformers[layer].feedforward.W_down.weight, params[layer_name + "mlp.down_proj.weight"]
        )
        qwen.transformers[layer].feedforward.W_gate.weight = assign(
            qwen.transformers[layer].feedforward.W_gate.weight, params[layer_name + "mlp.gate_proj.weight"]
        )
        qwen.transformers[layer].feedforward.W_up.weight = assign(
            qwen.transformers[layer].feedforward.W_up.weight, params[layer_name + "mlp.up_proj.weight"]
        )

        qwen.transformers[layer].norm2.weight = assign(
            qwen.transformers[layer].norm2.weight, params[layer_name + "post_attention_layernorm.weight"]
        )

        qwen.transformers[layer].attention.W_key.weight = assign(
            qwen.transformers[layer].attention.W_key.weight, params[layer_name + "self_attn.k_proj.weight"]
        )
        qwen.transformers[layer].attention.W_key.bias = assign(
            qwen.transformers[layer].attention.W_key.bias, params[layer_name + "self_attn.k_proj.bias"]
        )
        qwen.transformers[layer].attention.W_query.weight = assign(
            qwen.transformers[layer].attention.W_query.weight, params[layer_name + "self_attn.q_proj.weight"]
        )
        qwen.transformers[layer].attention.W_query.bias = assign(
            qwen.transformers[layer].attention.W_query.bias, params[layer_name + "self_attn.q_proj.bias"]
        )
        qwen.transformers[layer].attention.W_value.weight = assign(
            qwen.transformers[layer].attention.W_value.weight, params[layer_name + "self_attn.v_proj.weight"]
        )
        qwen.transformers[layer].attention.W_value.bias = assign(
            qwen.transformers[layer].attention.W_value.bias, params[layer_name + "self_attn.v_proj.bias"]
        )
        qwen.transformers[layer].attention.out_proj.weight = assign(
            qwen.transformers[layer].attention.out_proj.weight, params[layer_name + "self_attn.o_proj.weight"]
        )

    qwen.final_norm.weight = assign(
        qwen.final_norm.weight, params["model.norm.weight"]
    )

    torch.save(qwen.state_dict(), fdst)
    print("Successfully transform weights!")
