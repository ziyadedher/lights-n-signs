#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright © 2020 nishkrit <nishkrit@bionicdell>
#
# Distributed under terms of the MIT license.

"""
Simple script to extract the Waymo open dataset from TFRecord files to
tf.data.Example formats (or really anything that gives us more insight into
the dataset). The objective is to simply probe the dataset, so that we get
an understanding of what we are getting ourselves into.
"""

import os
import argparse

import tensorflow as tf
# tf.logging.set_verbosity(tf.logging.ERROR) # For neatness of output


def parse_dir_identifiers(parser) -> Tuple[str]:
    parser.add_argument('--data-dir', help='Path to Waymo open dataset directory')
    parser.add_argument('--data_env_variable', default='', required=False,
                        help='The name of the environment variable that is configured to point to the dataset directory')
    args = parser.parse_args()
    print(args)    

def get_dataset_files(data_dir: str, environ_var: str) -> List[str]:
    try:
        #TODO: Handle getting the path and extracting the file names
        pass
    except KeyError:
        raise EnvironmentError('Environment variable not found')
    except StopIteration:
        raise EnvironmentError('Invalid directory entered. Ensure that the folder exists')


def process_files(files):
    pass

def main():
    parser = argparse.ArgumentParser(prog="Waymoer")
    data_dir = parse_dir_identifiers(parser)
    files = get_dataset_files(data_dir)
    process_files(files)


if __name__=="__main__":
    main()
