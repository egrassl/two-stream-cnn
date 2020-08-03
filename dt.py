import argparse
import dataset_tools.data_transform as dt
import dataset_tools.data_split as ds

parser = argparse.ArgumentParser()

# Script arguments
parser.add_argument('command', metavar='command to execute', type=str, choices=['extract', 'split'],
                    help='command to execute')

# Parse arguments up to dst
args, remaining = parser.parse_known_args()

# Decides which command to run
if args.command == 'extract':
    parser.add_argument('src', metavar='dataset path', type=str, help='path to the RGB video dataset')
    parser.add_argument('dst', metavar='extraction destiny', type=str, help='path where extracted frames will be saved')
    parser.add_argument('--type', type=str, choices=['s', 't', 'st'],
                        help='indicates what kind of extraction will be done', required=True)
    parser.add_argument('--nb_frames', metavar='motion frames', type=int, help='number of motion frames per sample',
                        required=True)
    parser.add_argument('--chunks', metavar='chunks', type=int, help='how many samples per video', required=True)
    parser.add_argument('-v', default=False, action='store_true', help='verbose mode', required=False)
    args = parser.parse_args()

    dt.extract_from_dataset(args.src, args.dst, args.nb_frames, args.chunks, args.type == 's' or args.type == 'st',
                            args.type == 't' or args.type == 'st', args.v)

if args.command == 'split':
    parser.add_argument('src', metavar='dataset path', type=str, help='path to the RGB video dataset')
    parser.add_argument('dst', metavar='extraction destiny', type=str, help='path where extracted frames will be saved')
    parser.add_argument('mode', type=str, choices=['split', 'ntu-cs', 'ntu-cv'],
                        help='indicates what data modality will be moved be done')
    parser.add_argument('--type', type=str, choices=['s', 't', 'st'],
                        help='indicates what data modality will be moved be done', required=True)
    parser.add_argument('--val', metavar='motion frames', type=float, default=0, help='number of motion frames per sample',
                        required=False)
    parser.add_argument('--test', metavar='chunks', type=float, default=0, help='how many samples per video', required=False)
    parser.add_argument('-v', default=False, action='store_true', help='verbose mode', required=False)
    args = parser.parse_args()

    ds.split_dataset(args.src, args.dst, args.val, args.test, args.type == 's' or args.type == 'st',
                     args.type == 't' or args.type == 'st', mode=args.mode, verbose=args.v)


args = parser.parse_args()

# Executes extraction

