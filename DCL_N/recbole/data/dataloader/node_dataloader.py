# @Time   : 2020/9/23
# @Author : Yushuo Chen
# @Email  : chenyushuo@ruc.edu.cn

# UPDATE
# @Time   : 2022/7/8, 2020/9/23, 2020/12/28
# @Author : Zhen Tian, Yushuo Chen, Xingyu Pan
# @email  : chenyuwuxinn@gmail.com, chenyushuo@ruc.edu.cn, panxy@ruc.edu.cn

"""
recbole.data.dataloader.user_dataloader
################################################
"""
import torch
from logging import getLogger
from recbole.data.dataloader.abstract_dataloader import AbstractDataLoader
from recbole.data.interaction import Interaction
import numpy as np


class NodeDataLoader(AbstractDataLoader):
    """:class:`UserDataLoader` will return a batch of data which only contains user-id when it is iterated.

    Args:
        config (Config): The config of dataloader.
        dataset (Dataset): The dataset of dataloader.
        sampler (Sampler): The sampler of dataloader.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.

    Attributes:
        shuffle (bool): Whether the dataloader will be shuffle after a round.
            However, in :class:`UserDataLoader`, it's guaranteed to be ``True``.
    """

    def __init__(self, config, dataset, sampler, shuffle=False):
        self.logger = getLogger()
        if shuffle is False:
            shuffle = True
            self.logger.warning("UserDataLoader must shuffle the data.")

        self.nid_field = 'node_id'
        uid_tensor = torch.tensor(list(set(dataset['user_id'].tolist())))
        iid_tensor = torch.tensor(list(set(dataset['item_id'].tolist()))) + dataset.user_num
        nid_tensor = torch.cat((uid_tensor, iid_tensor))
        self.node_list = Interaction({self.nid_field: nid_tensor})
        self.sample_size = len(self.node_list)
        super().__init__(config, dataset, sampler, shuffle=shuffle)

    def _init_batch_size_and_step(self):
        batch_size = self.config["train_batch_size"]
        self.step = batch_size
        self.set_batch_size(batch_size)

    def collate_fn(self, index):
        index = np.array(index)
        return self.node_list[index]
    

    
