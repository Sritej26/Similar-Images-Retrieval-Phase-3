##################################################################################
# Authors: Ayush Anand, Pritam De, Sairaj Menon, Sritej Reddy,
#          Aaron Steele, Shubham Verma
# Course: CSE 515 Fall 2021, Arizona State University
# Professor: K. Selçuk Candan
# Project: Course Project Phase 3
# File: print_index_file.py
# 
# Description: Helper file used to print the contents of an index file.
##################################################################################

import argparse
from lib import latent
import pickle

def main(index_file):
    print(index_file)
    unprocessed_index_file = pickle.load(open(index_file, "rb"))
    if unprocessed_index_file[0] == 'lsh':
        index_method = 0
        [_, layers, hashes_per_layer, lsh, lsh_conversion_vector, image_dict, model, input_folder, dim_red_technique, results] = unprocessed_index_file
        print("LSH index file detected!")
        print(f"This index file was created with the following parameters:")
        print(f"\tNumber of Layers: {layers}")
        print(f"\tHashes per Layer: {hashes_per_layer}")
        print("\n")
        print(lsh, lsh_conversion_vector)
    # If the data is for a VA File
    elif unprocessed_index_file[0] == 'va':
        index_method = 1
        [_, bits, ff, pp, image_dict, model, folder_path, dim_red_technique, results] = unprocessed_index_file
        print("VA File detected!")
        print(f"This index file was created with the following parameters:")
        print(f"\tNumber of Bits per Dimension: {bits}")
        print("\n")
        print(ff, pp)

def parse_args():
    """
    Parse the input arguments for the program.
    """
    # Parsing of input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--index_file",         required=True,  help="Input image database folder")
    arguments = parser.parse_args()

    # If a latent semantic file is specified, we cannot use certain options
    return arguments.index_file

if __name__ == "__main__":
    main(parse_args())
