import json
import argparse
import pickle
import sys
import glob
import os


def merge_dict(dict1, dict2):
    ''' Merge dictionaries and keep values of common keys in list'''
    dict3 = {**dict1, **dict2}
    for key, value in dict3.items():
        if key in dict1 and key in dict2:
            dict3[key] = value + dict1[key]

    return dict3


def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--dict_dir',
        dest='dict_dir',
        default=None,
        type=str
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    detection_json_files = ""

    json_files = glob.glob(os.path.join(args.dict_dir, "*.json"))
    image_hash_map = {}
    for i, json_file in enumerate(json_files):
        tmp_hash_map = json.load(open(json_file, "r"))
        image_hash_map = merge_dict(image_hash_map, tmp_hash_map)

    json.dump(image_hash_map, open(os.path.join(args.dict_dir, "merged_dict.json"), "w+"))
