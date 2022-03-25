##################################################################################
# Authors: Ayush Anand, Pritam De, Sairaj Menon, Sritej Reddy,
#          Aaron Steele, Shubham Verma
# Course: CSE 515 Fall 2021, Arizona State University
# Professor: K. Selçuk Candan
# Project: Course Project Phase 3
# File: create_latent_file.py
# 
# Description: Helper file used to create latent semantic files if necessary.
##################################################################################

import argparse
from lib import latent

def main(*args):
    input_folder, model, dim_red_technique, k, output_folder = args
    image_dict, results = latent.compute_latent_semantics(input_folder, model, dim_red_technique, k, output_folder, True)
    print(f"Latent semantic file saved to '{output_folder}'!")

def parse_args():
    """
    Parse the input arguments for the program.
    """
    # Parsing of input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_folder",         required=True,  help="Input image database folder")
    parser.add_argument("-m", "--model",                required=True,  help="Model to compute feature vector")
    parser.add_argument("-d", "--dim_red_technique",    required=True,  help="Technique used to reduce the number of dimensions")
    parser.add_argument("-k", "--k",                    required=True,  help="Number of latent semantics to select")
    parser.add_argument("-o", "--output_folder",        required=True,  help="Output folder to save results")

    arguments = parser.parse_args()

    # If a latent semantic file is specified, we cannot use certain options
    return arguments.input_folder, arguments.model, arguments.dim_red_technique, arguments.k, arguments.output_folder

if __name__ == "__main__":
    main(*parse_args())
