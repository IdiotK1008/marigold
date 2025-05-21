import argparse
import os
import shutil

dir = 'data/hypersim/raw_data'
for dirpath, dirnames, filenames in os.walk(dir):
    for dirname in dirnames:
        dir_to_remove = os.path.join(dir, dirname, 'images')
        print(dir_to_remove)
        if os.path.exists(dir_to_remove):
            shutil.rmtree(dir_to_remove)
        # break
    # break