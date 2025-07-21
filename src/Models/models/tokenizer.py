# -*- coding: UTF-8 -*-

# 分词器类

from abc import ABC, abstractmethod
from tokenizers import Tokenizer as _ThirdTokenizer


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
    def __init__(self, file_path: str):
        self.tokenizer = _ThirdTokenizer.from_file(file_path)
        self._EOS_token = "<|end_of_sentence|>"
        self.tokenizer.add_special_tokens([self._EOS_token])
        self._EOS_id = self.tokenizer.get_vocab()[self._EOS_token]

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

    def eos_token(self) -> str:
        return self._EOS_token

    def eos_id(self) -> int:
        return self._EOS_id
