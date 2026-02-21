"""Allow running as: python -m parallel_mc"""
from .cli import main
import sys

if __name__ == '__main__':
    sys.exit(main())
