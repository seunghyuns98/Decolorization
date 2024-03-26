#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import json

from loss.losses import get_loss_list

losses_list = get_loss_list()

def args_parser():

    parser = argparse.ArgumentParser(description='Decolorization')

    # Mode selection
    parser.add_argument('--args', '-a', type=str, default='', choices=['defaults', 'json'],
                        help='How to read args? (json file or dataset defaults)')

    parser.add_argument('--exp', '--e', help='Experiment name')
    
    parser.add_argument('--workspace', '--ws', default='', type=str, help='Workspace name') 
    
    parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                        help='Path to latest checkpoint (default: none)')

    parser.add_argument('--evaluate', '--eval',  type=str, default=None,
                        help='Path to the checkpoint to evaluate.')

########### General Dataset arguments ###########

    parser.add_argument('--dataset-path',  type=str, default='', help='Dataset path.')

    parser.add_argument('--workers', default=4, type=int,
                        help='Number of dataloader workers (default: 4)')


########### Training arguments ###########
    parser.add_argument('--epochs', default=20, type=int,
                        help='Total number of epochs to run (default: 30)')
    
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,       
                        metavar='LR', help='Initial learning rate (default 0.001)')
    
    parser.add_argument('--loss', '-l', default='l1', choices=losses_list,
                        help='Loss function: ' + ' | '.join(losses_list) + ' (default: l1)')

########### Logging ###########

    parser.add_argument('--tb_log', default=False, type=bool,
                        help='Log to Tensorboard (default: False)')

    parser.add_argument('--tb_freq', default=1000, type=int,
                        help='Logging Frequence to Tensorboard (default: 1000)')


    # Parse the arguments
    args = parser.parse_args()

    args = initialize_args(args)

    return args


def initialize_args(args):
    # Check the "defaults" argument
    if args.args == 'defaults':  # Load the default parameters for the selected dataset
        args_path = 'dataloaders/datasets_defaults/' + args.dataset + '_defaults.json'
        load_args_from_file(args_path, args)
    elif args.args == 'json':
        # Path to the workspace directory
        ws_path = os.path.join('workspace/', args.workspace, args.exp)
        args_path = os.path.join(ws_path, 'args.json')
        load_args_from_file(args_path, args)
    return args


def save_args(exp_dir, args, file_name='args.json'):
    with open(os.path.join(exp_dir, file_name), 'w') as outfile:
        dic = {}
        for arg in vars(args):
            dic[arg] = getattr(args, arg)
        json.dump(dic, outfile, separators=(',\n', ': '))


# Load default args for each dataset
def load_args_from_file(args_file_path, given_args):

    if os.path.isfile(args_file_path):
        with open(args_file_path, 'r') as fp:
            loaded_args = json.load(fp)

        # Replace given_args with the loaded default values
        for key, value in loaded_args.items():
            if key not in ['workspace', 'exp', 'evaluate', 'resume']:  # Do not overwrite these keys
                setattr(given_args, key, value)

        print('\n==> Args were loaded from file "{}".'.format(args_file_path))
    else:
        print('\n==> Args file "{}" was not found!'.format(args_file_path))


# TODO: Print in a nicer way
def print_args(args):
    print('==> Experiment Args: \n {} \n'.format(args))


# This function compares the args saved in the checkpoint with the json file
def compare_args_w_json(chkpt_args, exp_dir, epoch):
    path_to_json = os.path.join(exp_dir, 'args.json')

    if os.path.isfile(path_to_json):
        with open(path_to_json, 'r') as fp:
            json_args = json.load(fp)

    old_args_saved = False
    for key, json_value in json_args.items():
        chkpt_value = getattr(chkpt_args, key)
        if chkpt_value != json_value:
            print('! Argument "{}" was changed from "{}" in the checkpoint to "{}" in the JSON file!'.format(key, chkpt_value, json_value))

            # Save the old args to another file for history
            #if not old_args_saved:
            #    f_name = 'args_epoch_' + str(epoch - 1) + '.json'
            #    save_args(exp_dir, chkpt_args, file_name=f_name)  # Save the original args
            #    print('==> Original args were saved to "{}".'.format(f_name))
            #    old_args_saved = True

            setattr(chkpt_args, key, json_value)

    print('')


if __name__ == '__main__':
    args_parser()
