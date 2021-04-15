import argparse
import logging
import math
import dill
import os
import options
import random
import numpy as np
from collections import OrderedDict

import torch
from torch import cuda
from torch.autograd import Variable

import data
import utils
from meters import AverageMeter
from generator import LSTMModel
from train_generator import train_g
from discriminator import Discriminator
from train import train_discriminator



logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)

parser = argparse.ArgumentParser(description="Driver program for JHU Adversarial-NMT.")

# Load args
options.add_general_args(parser)
options.add_dataset_args(parser)
options.add_distributed_training_args(parser)
options.add_optimization_args(parser)
options.add_checkpoint_args(parser)
options.add_generator_model_args(parser)
options.add_discriminator_model_args(parser)
options.add_generation_args(parser)

def main(args):
    use_cuda = (len(args.gpuid) >= 1)
    print("{0} GPU(s) are available".format(cuda.device_count()))

    print("======printing args========")
    print(args)
    print("=================================")

    # Load dataset
    splits = ['train', 'valid']
    if data.has_binary_files(args.data, splits):
        print("Loading bin dataset")
        dataset = data.load_dataset(
            args.data, splits, args.src_lang, args.trg_lang, args.fixed_max_len)
            #args.data, splits, args.src_lang, args.trg_lang)
    else:
        print(f"Loading raw text dataset {args.data}")
        dataset = data.load_raw_text_dataset(
            args.data, splits, args.src_lang, args.trg_lang, args.fixed_max_len)
            #args.data, splits, args.src_lang, args.trg_lang)
    if args.src_lang is None or args.trg_lang is None:
        # record inferred languages in args, so that it's saved in checkpoints
        args.src_lang, args.trg_lang = dataset.src, dataset.dst
    print('| [{}] dictionary: {} types'.format(dataset.src, len(dataset.src_dict)))
    print('| [{}] dictionary: {} types'.format(dataset.dst, len(dataset.dst_dict)))
    for split in splits:
        print('| {} {} {} examples'.format(args.data, split, len(dataset.splits[split])))

    g_logging_meters = OrderedDict()
    g_logging_meters['train_loss'] = AverageMeter()
    g_logging_meters['valid_loss'] = AverageMeter()
    g_logging_meters['train_acc'] = AverageMeter()
    g_logging_meters['valid_acc'] = AverageMeter()
    g_logging_meters['bsz'] = AverageMeter()  # sentences per batch

    d_logging_meters = OrderedDict()
    d_logging_meters['train_loss'] = AverageMeter()
    d_logging_meters['valid_loss'] = AverageMeter()
    d_logging_meters['train_acc'] = AverageMeter()
    d_logging_meters['valid_acc'] = AverageMeter()
    d_logging_meters['bsz'] = AverageMeter()  # sentences per batch

    # Set model parameters
    args.encoder_embed_dim = 1000
    args.encoder_layers = 4
    args.encoder_dropout_out = 0
    args.decoder_embed_dim = 1000
    args.decoder_layers = 4
    args.decoder_out_embed_dim = 1000
    args.decoder_dropout_out = 0
    args.bidirectional = False

    # try to load generator model
    g_model_path = 'checkpoints/generator/best_gmodel.pt'
    if not os.path.exists(g_model_path):
        print("Start training generator!")
        train_g(args, dataset)
    assert os.path.exists(g_model_path)
    generator = LSTMModel(args, dataset.src_dict, dataset.dst_dict, use_cuda=use_cuda)
    model_dict = generator.state_dict()
    pretrained_dict = torch.load(g_model_path)
    #print(f"First dict: {pretrained_dict}")
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    #print(f"Second dict: {pretrained_dict}")
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    #print(f"model dict: {model_dict}")
    # 3. load the new state dict
    generator.load_state_dict(model_dict)

    print("Generator has successfully loaded!")

    # try to load discriminator model
    d_model_path = 'checkpoints/discriminator/best_dmodel.pt'
    if not os.path.exists(d_model_path):
        print("Start training discriminator!")
        train_d(args, dataset)
    assert  os.path.exists(d_model_path)
    discriminator = Discriminator(args, dataset.src_dict, dataset.dst_dict, use_cuda=use_cuda)
    model_dict = discriminator.state_dict()
    pretrained_dict = torch.load(d_model_path)
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    discriminator.load_state_dict(model_dict)

    print("Discriminator has successfully loaded!")

    train_discriminator(dataset, generator, discriminator, g_logging_meters, d_logging_meters)

if __name__ == "__main__":
  ret = parser.parse_known_args()
  options = ret[0]
  if ret[1]:
    logging.warning("unknown arguments: {0}".format(parser.parse_known_args()[1]))
  main(options)