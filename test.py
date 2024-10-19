'''
Author: Yuchen Shi
Date: 20-10-2024 01:00:13
Last Editors: error: git config user.name & please set dead value or install git
Contact Last Editors: error: git config user.email & please set dead value or install git
LastEditTime: 20-10-2024 01:00:20
'''

import torch
print("是否可用：", torch.cuda.is_available())        # 查看GPU是否可用
print("GPU数量：", torch.cuda.device_count())        # 查看GPU数量
print("torch方法查看CUDA版本：", torch.version.cuda)  # torch方法查看CUDA版本
print("GPU索引号：", torch.cuda.current_device())    # 查看GPU索引号
print("GPU名称：", torch.cuda.get_device_name(0))    # 根据索引号得到GPU名称
