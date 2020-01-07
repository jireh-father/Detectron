from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os
import sys

import json

from itertools import islice


def chunks(data, SIZE=10000):
    it = iter(data)
    for i in range(0, len(data), SIZE):
        yield {k: data[k] for k in islice(it, SIZE)}


def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--image_hash_dict_file',
        dest='image_hash_dict_file',
        help='image_hash_dict_file',
        type=str
    )
    parser.add_argument(
        '--num_processes',
        dest='num_processes',
        help='num_processes',
        default=4,
        type=int
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    image_hash_dict = json.load(open(args.image_hash_dict_file))
    num_chunks = len(image_hash_dict) // args.num_processes
    hash_dict_chunks = list(chunks(image_hash_dict, num_chunks))
    if len(hash_dict_chunks) > args.num_processes:
        hash_dict_chunks[0].update(hash_dict_chunks[-1])
        hash_dict_chunks = hash_dict_chunks[:-1]

    output_dir = os.path.dirname(args.image_hash_dict_file)
    for i, hash_dict_chunk in enumerate(hash_dict_chunks):
        json.dump(hash_dict_chunk, open(os.path.join(output_dir, "image_hash_dict_chunk_%d.json" % i), "w+"))
