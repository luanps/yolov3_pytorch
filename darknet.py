from future import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def parse_cfg(cfgfile):
    """
    Takes a config file and returns a list of blocks
    (as a dictionary)
    These blocks are the neural network configurations
    """

    file = open(cfgfile, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if len(x) > 0]
    lines = [x for x in lines if x[0] != '#']
    lines = [x.rstrip().lstrip() for x in lines]

    block = {}
    blocks = []

    for line in lines:
        if line[0] =="[":   #new block
            if len(block) !=0: 
                blocks.append(block)
                block = {}
            block["type"] = line[1:-1].rstrip()
        else:
            key,value = line.split("=")
            block[key.rstrip()] = value.strip()
    blocks.append(block)

    return blocks

def create_modules(blocks):
    net_info = blocks[0]    #input info and pre-proessing
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []
    
    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()

        #check the type of block
        #create a new module 
        #append to module_list

        if (x["type"] == "convolutional"):
            activation = x["activation"]
            try:
                batch_norm = int(x["batch_normalize"])
                bias = False
            except:
                batch_norm = 0
                bias = True
        
            filters = int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])

            if padding:
                pad = (kernel_size -1)//2
            else:
                pad = 0

            #Add convoutional layer
            conv = nn.Conv2d(prev_filters,filters,kernel_size,...
                stride, pad, bias = bias)
            module.add_module("conv_{0}".format(index), conv)

            #Add batch norm layer
            if batch_norm:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".fomat(index),bn)

            #Check the activation
            if activation = "leaky":
                activn = nn.LeakyReLU(0.1, inplace = True)
                module.add_module("leaky_{0}".format(index), activn)
   
        #If it's an umsampling layer 
        elif (x["type"] == "upsample"):
            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor = 2, mode = "bilinear")
            module.add_module("upsample_{}".format(index), upsample)
            
        #It it's a route layer
        elif (x["type"] == "route"):
            x["layers"] = x["layers"].split(',')
            start = int(x["layers"][0])
            try:
                end = int(x["layers"][1])
            except:
                end = 0
            if start > 0 :
                start = start - index
            if end > 0:
                end = end - index
            route  EmptyLayer()
            module.add_module("route_{0}".format(index), route)
            if end < 0:
                filters = output_filters[index + start] + ,...
                    output_filters[index + end]
            else:
                filters = output_filters[index + start]

        #skip connection
        elif x["type"] == "shortcut":
            shortcut = EmptyLayer()
            module.add_module("shrtcut_{}".format(index), shortcut)
