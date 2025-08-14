# -*- coding: UTF-8 -*-

# 分词器类

from abc import ABC, abstractmethod
from tokenizers import Tokenizer as _ThirdTokenizer
import tiktoken


class Tokenizer(ABC):
    """
    分词器的抽象基类。所有分词器类都必须继承这个类，并实现这个类的所有方法
    """
    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """
        将文本编码为ID序列
        :param text:
        :return:
        """
        ...

    @abstractmethod
    def decode(self, ids: list[int]) -> str:
        """
        将ID序列解码成文本
        :param ids:
        :return:
        """
        ...

    @abstractmethod
    def split(self, text: str) -> list[str]:
        """
        分割文本
        :param text:
        :return:
        """
        ...

    @abstractmethod
    def look_up(self, token_id: int) -> str:
        """
        查找指定token_id的文本
        :param token_id:
        :return:
        """
        ...

    @abstractmethod
    def __len__(self) -> int:
        """
        返回词表大小
        :return:
        """
        ...

    @property
    def eos_id(self) -> int:
        raise NotImplementedError("No EOS token")

    @property
    def eos_token(self) -> str:
        raise NotImplementedError("No EOS token")


class JsonTokenizer(Tokenizer):
    def __init__(self, file_path: str, eos_id=-1):
        self.tokenizer = _ThirdTokenizer.from_file(file_path)
        if eos_id == -1:  # 自动推导eos
            self._EOS_token = "<|end_of_sentence|>"
            self.tokenizer.add_special_tokens([self._EOS_token])
            self._EOS_id = self.tokenizer.get_vocab()[self._EOS_token]
        else:  # 指定eos
            self._EOS_id = eos_id
            self._EOS_token = self.tokenizer.decode([eos_id], skip_special_tokens=False)

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, ids: list[int], skip_special=True) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special)

    def split(self, text: str, skip_special=True) -> list[str]:
        result = []
        ids = self.tokenizer.encode(text).ids
        for i in ids:
            result.append(self.tokenizer.decode([i], skip_special_tokens=skip_special))
        return result

    def look_up(self, token_id: int) -> str:
        return self.tokenizer.decode(token_id)

    def __len__(self):
        return self.tokenizer.get_vocab_size()

    @property
    def eos_token(self) -> str:
        return self._EOS_token

    @property
    def eos_id(self) -> int:
        return self._EOS_id


class TikTokenizer(Tokenizer):
    def __init__(self, model_name: str):
        self.tokenizer = tiktoken.get_encoding(model_name)

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)

    def split(self, text: str) -> list[str]:
        result = []
        ids = self.tokenizer.encode(text)
        for i in ids:
            result.append(self.tokenizer.decode([i]))
        return result

    def look_up(self, token_id: int) -> str:
        return self.tokenizer.decode([token_id])

    def __len__(self):
        return self.tokenizer.n_vocab

    @property
    def eos_token(self) -> str:
        return self.look_up(self.tokenizer.eot_token)

    @property
    def eos_id(self) -> int:
        return self.tokenizer.eot_token
