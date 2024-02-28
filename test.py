
from main import main
import argparse
# 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19
for ms in [0]:
    args = argparse.Namespace(myseed= ms, dataset='CIFAR10', method='TopOne')
    main(args)
