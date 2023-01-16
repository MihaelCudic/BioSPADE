"""
Functions sourced from https://github.com/NVlabs/SPADE and adjusted accordingly for bioSPADE
"""


import torch
from models.networks.base_network import BaseNetwork
from models.networks.loss import *
from models.networks.discriminator import *
from models.networks.generator import *
from models.networks.segmentor import *
import util.util as util


def find_network_using_name(target_network_name, filename):
    target_class_name = target_network_name + filename
    module_name = 'models.networks.' + filename
    network = util.find_class_in_module(target_class_name, module_name)

    assert issubclass(network, BaseNetwork), \
        "Class %s should be a subclass of BaseNetwork" % network

    return network


def modify_commandline_options(parser, is_train):
    opt, _ = parser.parse_known_args()

    netG_cls = find_network_using_name(opt.netG, 'generator')
    parser = netG_cls.modify_commandline_options(parser, is_train)
    if is_train:
        netD_cls = find_network_using_name(opt.netD, 'discriminator')
        parser = netD_cls.modify_commandline_options(parser, is_train)
    netE_cls = find_network_using_name('conv', 'encoder')
    parser = netE_cls.modify_commandline_options(parser, is_train)

    return parser


def create_network(cls, opt, network):
    net = cls(opt, network)
    net.print_network()
    if len(opt.gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.cuda()
    net.init_weights(opt.init_type, opt.init_variance)
    return net

def define_model(opt, model):
    model_type = None
    if 'G'==model[0]:
        model_type = 'generator'
    elif 'D'==model[0]:
        model_type = 'discriminator'
    elif 'S'==model[0]:
        model_type = 'segmentor'
    
    net = find_network_using_name(getattr(opt, 'net_'+model), model_type)
    return create_network(net, opt, model)