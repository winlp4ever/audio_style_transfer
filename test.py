import argparse

prs = argparse.ArgumentParser()
prs.add_argument('--sr', '--sampling', help='', type=int, nargs='?', default=0)

args = prs.parse_args()

print(args.sampling)