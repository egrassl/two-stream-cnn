import argparse
import dataset_tools.data_transform as dt
import dataset_tools.ntu_definitions as ntu
import random

parser = argparse.ArgumentParser()

# Script arguments
parser.add_argument('mode', type=str, choices=['ntu-cs', 'ntu-cv', 'split'], help='Indicates what split will be done'),
parser.add_argument('src', metavar='dataset path', type=str, help='path to the RGB video dataset')
parser.add_argument('dst', metavar='extraction destiny', type=str, help='path where extracted frames will be saved')
parser.add_argument('--type', type=str, choices=['s', 't', 'st'],  help='indicates what kind of extraction will be done', required=True)
parser.add_argument('--nb_frames', metavar='motion frames', type=int, help='number of motion frames per sample', required=True)
parser.add_argument('--chunks', metavar='chunks', type=int, help='how many samples per video', required=True)
parser.add_argument('-v', default=False, action='store_true', help='verbose mode', required=False)
args = parser.parse_args()


# Executes extraction
spatial = args.type == 's' or args.type == 'st'
temporal = args.type == 't' or args.type == 'st'

if args.mode == 'ntu-cs':
    split_func = ntu.get_cs_split
elif args.mode == 'ntu-cv':
    split_func = ntu.get_cv_split
else:
    split_func = lambda x: 'train' if random.uniform(0, 1) <= .7 else 'val'

extractor = dt.DataExtract(src=args.src,
                           dst=args.dst,
                           nb_frames=args.nb_frames,
                           chunks=args.chunks,
                           split_func=split_func,
                           spatial=spatial,
                           temporal=temporal,
                           verbose=args.v)

extractor.extract()
