"""
Functions sourced from https://github.com/NVlabs/SPADE and adjusted for bioSPADE
"""

import torch.utils.data as data

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        pass