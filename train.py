
import argparse
import dill
import logging
import numpy as np
import math
from collections import OrderedDict

import torch
import torch.nn as nn
from torch import cuda
from torch.autograd import Variable
import torch.nn.functional as F


import options
import data
from meters import AverageMeter
from model import NMT


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
options.add_model_args(parser)

def main(args):
    use_cuda = (len(args.gpuid) >= 1)
    if args.gpuid:
        cuda.set_device(args.gpuid[0])

    # Load dataset
    splits = ['train', 'valid']
    if data.has_binary_files(args.data, splits):
        dataset = data.load_dataset(
            args.data, splits, args.src_lang, args.trg_lang)
    else:
        dataset = data.load_raw_text_dataset(
            args.data, splits, args.src_lang, args.trg_lang)
    if args.src_lang is None or args.trg_lang is None:
        # record inferred languages in args, so that it's saved in checkpoints
        args.src_lang, args.trg_lang = dataset.src, dataset.dst
    print('| [{}] dictionary: {} types'.format(dataset.src, len(dataset.src_dict)))
    print('| [{}] dictionary: {} types'.format(dataset.dst, len(dataset.dst_dict)))
    for split in splits:
        print('| {} {} {} examples'.format(args.data, split, len(dataset.splits[split])))

    # Set model parameters
    args.encoder_embed_dim = 1000
    args.encoder_layers = 4
    args.encoder_dropout_out = 0
    args.decoder_embed_dim = 1000
    args.decoder_layers = 4
    args.decoder_out_embed_dim = 1000
    args.decoder_dropout_out = 0
    args.bidirectional = False

    logging_meters = OrderedDict()
    logging_meters['train_loss'] = AverageMeter()
    logging_meters['valid_loss'] = AverageMeter()
    logging_meters['bsz'] = AverageMeter()  # sentences per batch

    # Build model
    generator = NMT(args, dataset.src_dict, dataset.dst_dict, use_cuda=use_cuda, is_testing=False)
    generator.decoder.is_testing = False

    if use_cuda:
        generator.cuda()
    else:
        generator.cpu()

    criterion = torch.nn.NLLLoss(size_average = False, ignore_index = dataset.dst_dict.pad(),
                                 reduce = True)

    optimizer = eval("torch.optim." + args.optimizer)(generator.parameters(), args.learning_rate)

    # main training loop
    for epoch_i in range(1, args.epochs + 1):
        logging.info("At {0}-th epoch.".format(epoch_i))

        seed = args.seed + epoch_i
        torch.manual_seed(seed)

        max_positions_train = (args.max_source_positions, args.max_target_positions)

        # Initialize dataloader, starting at batch_offset
        itr = dataset.train_dataloader(
            'train',
            max_tokens=args.max_tokens,
            max_sentences=args.max_sentences,
            max_positions=max_positions_train,
            seed=seed,
            epoch=epoch_i,
            sample_without_replacement=args.sample_without_replacement,
            sort_by_source_size=(epoch_i <= args.curriculum),
            shard_id=args.distributed_rank,
            num_shards=args.distributed_world_size,
        )
        # set training mode
        generator.train()

        # # update learning rate if necessary
        update_learning_rate(epoch_i, args.lr_shrink, args.lr_shrink_from, optimizer)

        # reset meters
        for key, val in logging_meters.items():
            if val is not None:
                val.reset()

        # srange generates a lazy sequence of shuffled range
        for i, sample in enumerate(itr):

            if use_cuda:
                sample['id'] = sample['id'].cuda()
                sample['net_input']['src_tokens'] = sample['net_input']['src_tokens'].cuda()
                sample['net_input']['src_lengths'] = sample['net_input']['src_lengths'].cuda()
                sample['net_input']['prev_output_tokens'] = sample['net_input']['prev_output_tokens'].cuda()
                sample['target'] = sample['target'].cuda()

            sys_out_batch, _ = generator(sample)  # (trg_seq_len, batch_size, trg_vocab_size)
            train_trg_batch = sample['target'].view(-1)
            sys_out_batch = sys_out_batch.contiguous().view(-1, sys_out_batch.size(-1))
            loss = criterion(sys_out_batch, train_trg_batch)
            sample_size = sample['target'].size(0) if args.sentence_avg else sample['ntokens']
            nsentences  = sample['target'].size(0)
            logging_loss = loss.data / sample_size / math.log(2)
            logging_meters['bsz'].update(nsentences)
            logging_meters['train_loss'].update(logging_loss, sample_size)
            logging.debug("loss at batch {0}: {1:.3f}, batch size: {2}".format(i, logging_meters['train_loss'].avg, round(logging_meters['bsz'].avg)))
            optimizer.zero_grad()
            loss.backward()

            # all-reduce grads and rescale by grad_denom
            for p in generator.parameters():
                if p.requires_grad:
                    p.grad.data.div_(sample_size)

            torch.nn.utils.clip_grad_norm(generator.parameters(), args.clip_norm)
            optimizer.step()

        # validation -- this is a crude estimation because there might be some padding at the end
        # Initialize dataloader
        max_positions_valid = (1e5, 1e5)

        itr = dataset.eval_dataloader(
            'valid',
            max_tokens=args.max_tokens,
            max_sentences=args.max_sentences,
            max_positions=max_positions_valid,
            skip_invalid_size_inputs_valid_test=args.skip_invalid_size_inputs_valid_test,
            descending=True,  # largest batch first to warm the caching allocator
            shard_id=args.distributed_rank,
            num_shards=args.distributed_world_size,
        )
        # set validation mode
        generator.eval()

        # reset meters
        for key, val in logging_meters.items():
            if val is not None:
                val.reset()

        for i, sample in enumerate(itr):
            with torch.no_grad():
                if use_cuda:
                    sample['id'] = sample['id'].cuda()
                    sample['net_input']['src_tokens'] = sample['net_input']['src_tokens'].cuda()
                    sample['net_input']['src_lengths'] = sample['net_input']['src_lengths'].cuda()
                    sample['net_input']['prev_output_tokens'] = sample['net_input']['prev_output_tokens'].cuda()
                    sample['target'] = sample['target'].cuda()
                sys_out_batch, _ = generator(sample)
                dev_trg_batch = sample['target'].view(-1)
                sys_out_batch = sys_out_batch.contiguous().view(-1, sys_out_batch.size(-1))
                loss = criterion(sys_out_batch, dev_trg_batch)
                sample_size = sample['target'].size(0) if args.sentence_avg else sample['ntokens']
                loss = loss / sample_size / math.log(2)
                logging_meters['valid_loss'].update(loss, sample_size)
                logging.debug("dev loss at batch {0}: {1:.3f}".format(i, logging_meters['valid_loss'].avg))


        logging.info("Average loss value per instance is {0} at the end of epoch {1}".format(logging_meters['valid_loss'].avg, epoch_i))

        torch.save(generator, open(args.model_file + ".nll_{0:.3f}.epoch_{1}".format(logging_meters['valid_loss'].avg, epoch_i), 'wb'), pickle_module=dill)


def update_learning_rate(current_epoch, lr_shrink, lr_shrink_from, optimizer):
    if (current_epoch == lr_shrink_from):
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_shrink

if __name__ == "__main__":
    ret = parser.parse_known_args()
    args = ret[0]
    if ret[1]:
        logging.warning("unknown arguments: {0}".format(parser.parse_known_args()[1]))
    main(args)
