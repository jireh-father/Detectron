import json
import pickle
import argparse
import sys
import os

def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--dict_file',
        dest='dict_file',
        default=None,
        type=str
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    dict_map = json.load(open(args.dict_file))
    pickle.dump(dict_map, open(os.path.splitext(args.dict_file)[0] + ".pkl", "wb+"))
