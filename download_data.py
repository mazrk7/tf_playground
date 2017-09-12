"""Download MNIST, Omniglot datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import urllib
import os

import config

OMNIGLOT_URL = 'https://github.com/yburda/iwae/raw/master/datasets/OMNIGLOT/chardata.mat'

if __name__ == '__main__':
    if not os.path.exists(config.DATA_DIR):
        os.makedirs(config.DATA_DIR)

    # Get Omniglot
    local_filename = os.path.join(config.DATA_DIR, config.OMNIGLOT)
    if not os.path.exists(local_filename):
        urllib.urlretrieve(OMNIGLOT_URL, local_filename)
