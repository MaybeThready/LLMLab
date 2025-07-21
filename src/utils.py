# -*- coding: UTF-8 -*-

# 在这里写入一些你认为调包时经常用到的小函数

import os


def get_abspath_from_dir(dir_path: str) -> list[str]:
    names = os.listdir(dir_path)
    paths = [os.path.join(os.path.abspath(dir_path), name) for name in names]
    return paths
