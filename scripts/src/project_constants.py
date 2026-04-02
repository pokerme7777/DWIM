import os
import argparse
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_CACHE_LOCAL = os.path.join(PROJECT_ROOT, 'cache')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("constant", help="Name of the constant to print", type=str, choices=globals().keys())
    args = parser.parse_args()
    constant_value = globals()[args.constant]
    sys.stdout.write(constant_value)
    sys.stdout.flush()