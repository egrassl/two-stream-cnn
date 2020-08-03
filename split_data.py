import argparse
import dataset_tools.data_split as ds

parser = argparse.ArgumentParser()

# Script arguments
parser.add_argument('--src', metavar='Dataset path', type=str, help='Path to the RGB video dataset', required=True)
parser.add_argument('--dst', metavar='Extraction destiny', type=str, help='Path where extracted frames will be saved', required=True)
parser.add_argument('--val', metavar='Motion frames', type=float, help='Number of motion frames per sample', required=True)
parser.add_argument('--test', metavar='Chunks', type=float, help='How many samples per video', required=True)
parser.add_argument('-v', default=False, action='store_true', help='Verbose mode', required=False)

args = parser.parse_args()

# Executes extraction
ds.split_spatial(args.src, args.dest, args.val, args.test, args.v)
