import argparse
import dataset_tools.data_transform as dt

parser = argparse.ArgumentParser()

# Script arguments
parser.add_argument('--src', metavar='Dataset path', type=str, help='Path to the RGB video dataset', required=True)
parser.add_argument('--dest', metavar='Extraction destiny', type=str, help='Path where extracted frames will be saved', required=True)
parser.add_argument('--nb_frames', metavar='Motion frames', type=int, help='Number of motion frames per sample', required=True)
parser.add_argument('--chunks', metavar='Chunks', type=int, help='How many samples per video', required=True)
parser.add_argument('-s', default=False, action='store_true', help='Indicates if spatial extraction will be done', required=False)
parser.add_argument('-t', default=False, action='store_true', help='Indicates if temporal extraction will be done', required=False)
parser.add_argument('-v', default=False, action='store_true', help='Verbose mode', required=False)

args = parser.parse_args()

# Executes extraction
dt.extract_from_dataset(args.src, args.dest, args.nb_frames, args.chunks, args.s, args.t, args.v)
