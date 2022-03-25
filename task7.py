##################################################################################
# Authors: Ayush Anand, Pritam De, Sairaj Menon, Sritej Reddy,
#          Aaron Steele, Shubham Verma
# Course: CSE 515 Fall 2021, Arizona State University
# Professor: K. Selçuk Candan
# Project: Course Project Phase 3
# File: task7.py
# 
# Description: Main driver file for Task 7 of the assignment.
##################################################################################

import argparse
from lib import io, latent, ppr, stats, util, svmBinary
from simple_term_menu import TerminalMenu
import os
import pickle
import numpy as np
import pandas as pd

def main(*args):
    index_file, query_result_file, output_folder = args
    
    # To test this file, run the following commands:
    # python3 task4.py -l 5 -k 10 -t 10 -q test_images/image-cc-19-5.png -m elbp -o Outputs -f phase3_data/1000 
    # python3 task7.py -i <output_index_file> -q <output_query_file>
    
    # First, we read a saved index file (generated in tasks 4-5) and query result file
    unprocessed_index_file = pickle.load(open(index_file, "rb"))
    if unprocessed_index_file[0] == 'lsh':
        index_method = 0
        [_, layers, hashes_per_layer, lsh, lsh_conversion_vector, image_dict, model, input_folder, dim_red_technique, results] = unprocessed_index_file
        print("LSH index file detected!")
        print(f"This index file was created with the following parameters:")
        print(f"\tNumber of Layers:    {layers}")
        print(f"\tHashes per Layer:    {hashes_per_layer}")
    # If the data is for a VA File
    elif unprocessed_index_file[0] == 'va':
        index_method = 1
        [_, bits, ff, pp, image_dict, model, folder_path, dim_red_technique, results] = unprocessed_index_file
        print("VA File detected!")
        print(f"This index file was created with the following parameters:")
        print(f"\tNumber of Bits per Dimension: {bits}")
    
    # For query relevance feedback, the relelvant parameters/variables are:
    #  - image_dict: This contains the images stored in the index file along with the original feature vectors
    #  - model: This specifies which type of feature vector to use. This will be 'color', 'elbp', 'hog', or 'latent'
    #  - query_results_imagenames: The names of the results of the query images. This is the set of labeled
    #           data for refinement.
    #  - user_labeled_tuples: Labeled tuples from user input.
    
    input_folder, query_image, query_image_name, processed_query_vector, tasknum, query_results, t = io.read_query_results(query_result_file)

    # First, put a terminal menu to allow the user to select relevant images.
    query_results_imagenames = [x[0] for x in query_results[:t]]
    io.display_query_results(image_dict, query_image, query_image_name, query_results_imagenames, output_folder, 'task7_temp_img.png')
    # A further loop to continually refine query results, if necessary.
    # We can remove this from the loop if there's no need
    user_labeled_tuples = []
    data_x = []
    data_y = []
    considered_imagenames = []
    while True:
        # Failsafe to ensure that we cannot 'over-refine'
        if len(considered_imagenames) > len(query_results) - t or len(query_results_imagenames) < t:
            print("Query can no longer be refined! Too few images are left!")
            break
        # We can also choose just one of these methods to refine results (or use one per method or something)
        query_refine_options = ["Yes", "No"]
        terminal_menu = TerminalMenu(query_refine_options, title="Refine query?")
        query_option_index = terminal_menu.show()
        
        if query_option_index == 1:
            break
        
        # The results are already displayed, so now we select which ones we think are relevant
        terminal_menu = TerminalMenu(
            query_results_imagenames,
            multi_select=True,
            show_multi_select_hint=True,
            title="Mark some images as 'Relevant':"
        )
        query_relevance_indices = terminal_menu.show()
        
        temp_images = []
        for i in range(len(query_results_imagenames)):
            if query_results_imagenames[i] not in considered_imagenames and i in query_relevance_indices:
                user_labeled_tuples.append((query_results_imagenames[i], 1))
                data_x.append(image_dict[query_results_imagenames[i]][model])
                data_y.append(1)
                considered_imagenames.append(query_results_imagenames[i])
            elif query_results_imagenames[i] not in considered_imagenames:
                temp_images.append(query_results_imagenames[i])
                
        terminal_menu = TerminalMenu(
            temp_images,
            multi_select=True,
            show_multi_select_hint=True,
            title="Mark some images as 'Irrelevant':"
        )
        query_irrelevance_indices = terminal_menu.show()
        
        for i in range(len(temp_images)):
            if temp_images[i] not in considered_imagenames and i in query_irrelevance_indices:
                user_labeled_tuples.append((query_results_imagenames[i], 0))
                data_x.append(image_dict[query_results_imagenames[i]][model])
                data_y.append(0)
                considered_imagenames.append(query_results_imagenames[i])
                
        # First, get a set of the feature vectors of all query images (relevant and irrelevant)
        print("Refining results with an SVM classifier...")
        
        W, labels = svmBinary.fit(data_x, data_y.copy())
        unlabeled_imagenames = [imgname[0] for imgname in query_results if imgname[0] not in considered_imagenames]
        unlabeled_vectors = [image_dict[x][model] for x in unlabeled_imagenames]
        y_predicted, y_d, labeled_distances = svmBinary.predict(unlabeled_vectors, W, data_x, labels)
        # Processing to extract the NEW top t relevant images based on this reordering
        # First, create tuples of all images labeled as 'relevant' (user-specified or not)
        relevant_tuples = []
        irrelevant_tuples = []
        for i in range(len(y_predicted)):
            if y_predicted[i] == 1:
                relevant_tuples.append((unlabeled_imagenames[i], y_d[i]))
            else:
                irrelevant_tuples.append((unlabeled_imagenames[i], y_d[i]))
        j = 0
        for tuple in user_labeled_tuples:
            if tuple[1] == 1:
                relevant_tuples.append((tuple[0], labeled_distances[j]))
            else:
                irrelevant_tuples.append((tuple[0], labeled_distances[j]))
            j += 1
        # Now, sort these in DECREASING order based on the distances
        relevant_tuples.sort(key=lambda x:x[1], reverse=True)
        relevant_imagenames = [x[0] for x in relevant_tuples][:t]
        io.display_query_results(image_dict, query_image, query_image_name, relevant_imagenames, output_folder, 'task7_img_results.png')
        query_results_imagenames = relevant_imagenames
    print('Goodbye!')
    
def parse_args():
    """
    Parse the input arguments for the program.
    """
    # Parsing of input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--index_file",    required=True, help="Input index file")
    parser.add_argument("-q", "--query_result_file", required=True, help="File containing results from a nearest neighbor query (run on the index file).")
    parser.add_argument("-o", "--output_folder",     required=True, help="Output folder for results")
    
    arguments = parser.parse_args()
    
    # Checking that specified input files/query results files are valid.
    if not os.path.exists(os.path.join(os.getcwd(), arguments.index_file)):
        parser.error("Input index file does not exist!")
    if not os.path.exists(os.path.join(os.getcwd(), arguments.query_result_file)):
        parser.error("Input query result file does not exist!")
    if not os.path.isdir(arguments.output_folder):
        parser.error("Output folder does not exist!")
    
    return arguments.index_file, arguments.query_result_file, arguments.output_folder

if __name__ == "__main__":
    main(*parse_args())