##################################################################################
# Authors: Ayush Anand, Pritam De, Sairaj Menon, Sritej Reddy,
#          Aaron Steele, Shubham Verma
# Course: CSE 515 Fall 2021, Arizona State University
# Professor: K. Selçuk Candan
# Project: Course Project Phase 3
# File: task8.py
# 
# Description: Main driver file for Task 8 of the assignment.
#
# This is different from other files in this project, as relatively few command
# line arguments are provided. However, interactive menus are used to get user
# input.
##################################################################################

import argparse
from simple_term_menu import TerminalMenu
from lib import io, feature, latent, LSH, util, svmBinary, va
import os
import pickle
import scipy.spatial
import numpy as np

# Decision Tree Imports
from lib.dt import DTClassifier
from sklearn.tree import DecisionTreeClassifier
import scipy.spatial.distance

def main(*args):
    input_folder, index_file, output_folder = args
    
    # Either read the index file or create one from the input_folder
    if index_file is None:
        print(f"No index file specified! We need to create an index file on the images in '{input_folder}'!\n")
        options = ["Locality Sensitive Hashing", "VA File"]
        terminal_menu = TerminalMenu(options, title="Type of index file:")
        index_method = terminal_menu.show()
        
        # Case when LSH chosen
        if index_method == 0:
            layers = io.get_int_from_user("Number of layers in LSH: ")
            hashes_per_layer = io.get_int_from_user("Number of hashes per layer for LSH: ")
        # Case when VA File chosen
        else:
            bits = io.get_int_from_user("Number of bits per dimension: ")
        
        vector_options = ["Color Moments", "ELBP", "HOG"]
        terminal_menu = TerminalMenu(vector_options, title="Type of image vector:")
        vector_method = terminal_menu.show()
        
        if vector_method == 0:
            model = 'color'
            orig_model = 'color'
        elif vector_method == 1:
            model = 'elbp'
            orig_model = 'elbp'
        else:
            model = 'hog'
            orig_model = 'hog'
        
        latent_bool_options = ["Yes", "No"]
        terminal_menu = TerminalMenu(latent_bool_options, title="Create and use latent features?")
        latent_bool = terminal_menu.show()
        
        # Additional options if we use latent features
        if latent_bool == 0:
            latent_extraction_options = ["PCA", "SVD"]
            terminal_menu = TerminalMenu(latent_extraction_options, title="Method of feature extraction:")
            latent_option = terminal_menu.show()
            if latent_option == 0:
                dim_red_technique = 'pca'
            elif latent_option == 1:
                dim_red_technique = 'svd'
            k = io.get_int_from_user("Number of latent features to extract: ")
            
        print(f"\nReading in images from {input_folder} and computing features of type {vector_options[vector_method]}...")
        
        # The only difference is that in the first method we do not bother computing latent vectors
        if latent_bool != 0:
            image_dict = io.read_images(input_folder)
            feature_len = feature.compute_features(image_dict, model)
        else:
            print("Computing latent vectors...")
            image_dict, results = latent.compute_latent_semantics(input_folder, model, dim_red_technique, k, output_folder, False)
        
        print("Done!")
        
        # Set model to 'latent' if we use latent features. This is helpful down the line
        if latent_bool == 0:
            model = 'latent'
        
        if index_method == 0:
            print("Creating LSH index file on input images...")
            data_array = util.feature_dict_to_array(image_dict, model)
            lsh, lsh_conversion_vector = LSH.create_base_LSH(hashes_per_layer, data_array, layers, image_dict)
            print("Done!")
        else:
            print("Creating VA index file on input images...")
            data_array = util.feature_dict_to_array(image_dict, model)
            ff, pp, approx, bytes_size = va.index_partition(data_array, bits)
            print("Done!")
    else:
        print(f"Reading index file from {index_file}...")
        unprocessed_index_file = pickle.load(open(index_file, "rb"))
        # If the data is for LSH
        if unprocessed_index_file[0] == 'lsh':
            index_method = 0
            [_, layers, hashes_per_layer, lsh, lsh_conversion_vector, image_dict, model, input_folder, dim_red_technique, results] = unprocessed_index_file
            print("LSH index file detected!")
            print(f"This index file was created with the following parameters:")
            print(f"\tInput Folder:     {input_folder}")
            print(f"\tNumber of Layers: {layers}")
            print(f"\tHashes per Layer: {hashes_per_layer}")
        # If the data is for a VA File
        elif unprocessed_index_file[0] == 'vafile':
            index_method = 1
            print("TODO!")
        # This branch should never trigger
        else:
            print("Error! Index file is invalid!")
            exit(1)
        print("Done reading in index file!")
    
    # Loop to continually get queries and allow the user to refine the result
    while True:
        # First, determine whether to get a new query image or quit
        options = ["New Query", "Quit"]
        terminal_menu = TerminalMenu(options, title="Options:")
        menu_entry_index = terminal_menu.show()
        
        # Only need to handle the case if we quit, since there's no other options. Change this if we
        # add another option for some reason.
        if menu_entry_index == 1:
            print("Exiting! Goodbye!")
            exit(0)
        
        # Loop until we get a valid filename
        print("\nPerforming new query!")
        valid_image_filename = False
        while(not valid_image_filename):
            query_image_filename = input("Input image filename: ")
            if not os.path.exists(os.path.join(os.getcwd(), query_image_filename)):
                print("Invalid image filename! Try again.")
            else:
                valid_image_filename = True
        
        t = io.get_int_from_user("Number of images to return in query: ")
        
        # Read in and process the query image
        query_image = io.read_image(query_image_filename)
        query_feature_vector = feature.compute_single_feature(query_image, orig_model)
        # If using latent features, map to the latent space
        if latent_bool == 0:
            query_feature_vector = latent.map_single_image_to_latent(query_feature_vector, dim_red_technique, results)
        # If the index file is LSH
        if index_method == 0:
            # Now, perform the query
            query = LSH.map_LSH_query(layers, lsh_conversion_vector, query_feature_vector, query_image_filename)
            query_results = LSH.get_t_most_similar(lsh, query, query_feature_vector, layers, t, image_dict, model)
            #unlabeled_imagenames = query_results[t:]
            query_results_imagenames = query_results[:t]
        # If the index file is a VA-File
        else:
            ans, _, _ = va.VA_SSA(data_array, ff, pp, query_feature_vector, t * 5)
            returned_imgnames = [list(image_dict.keys())[int(i)] for i in ans]
            query_results = []
            for img in returned_imgnames:
                query_results.append((img, scipy.spatial.distance.euclidean(query_feature_vector, image_dict[img][model])))
            # Now, sort the results and get the top t
            query_results.sort(key = lambda x: x[1])
            query_results = [x[0] for x in query_results]
            query_results_imagenames = query_results[:t]
        
        # If we don't break the loop, choose a classifier and use it to refine results
        classifier_options = ["Decision Tree", "SVM"]
        terminal_menu = TerminalMenu(classifier_options, title="Classifier to use:")
        classifier_index = terminal_menu.show()
        io.display_query_results(image_dict, query_image, query_image_filename, query_results_imagenames, output_folder, 'task8_temp_img.png')
        # A further loop to continually refine query results, if necessary.
        # We can remove this from the loop if there's no need
        user_labeled_tuples = []
        data_x = []
        data_y = []
        considered_imagenames = []
        while True:
            if len(considered_imagenames) > len(query_results) - t or len(query_results_imagenames) < t:
                print("Query can no longer be refined! Too few images are left!")
                print("Please enter another query, or re-do this one.")
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
            # Selected images is a list which contains the names of the selected entries
            # selected_images = list(terminal_menu.chosen_menu_entries)

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
        
            if classifier_index == 1:
                print("Refining results with an SVM classifier...")
                # First, get a set of the feature vectors of all query images (relevant and irrelevant)
                W, labels = svmBinary.fit(data_x, data_y.copy())
                unlabeled_imagenames = [imgname for imgname in query_results if imgname not in considered_imagenames]
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
            elif classifier_index == 0:
                print("Refining results with a Decision Tree classifier...")
                
                # Logic to use DT here to get new query results.
                data_x_arr = np.array(data_x)
                labels_arr = np.array([[el] for el in data_y])

                ## USING CUSTOM DT IMPLEMENTATION
                dt_clf = DTClassifier(min_samples_split=2, max_depth=5)
                dt_clf.fit(data_x_arr, labels_arr) 
                
                unlabeled_imagenames = [imgname for imgname in query_results if imgname not in considered_imagenames]
                unlabeled_vectors = [image_dict[x][model] for x in unlabeled_imagenames]
                unlabeled_vectors_arr = np.array(unlabeled_vectors)
                
                # Prediction part
                label_results = np.array(dt_clf.predict(unlabeled_vectors_arr)).astype(int)

                # Combining all the selected images in a dictionary
                #unlabeled_selected = prep_selected_image_dict(label_results, unlabeled_imagenames, unlabeled_vectors)
                # labeled_selected = prep_selected_image_dict(data_y, query_results_imagenames, data_x)
                # Merge both the relevant dictionary results
                #labeled_selected.update(unlabeled_selected)
                # First, create tuples of all images labeled as 'relevant' (user-specified or not)
                temp_dict = {}
                for i in range(len(label_results)):
                    if label_results[i] == 1:
                        temp_dict[unlabeled_imagenames[i]] = image_dict[unlabeled_imagenames[i]][model]
                j = 0
                for tuple in user_labeled_tuples:
                    if tuple[1] == 1:
                        temp_dict[tuple[0]] = image_dict[tuple[0]][model]
                    j += 1
                # Getting input image features based on the chosen model
                topImages = compute_vector_difference(temp_dict, query_feature_vector)
                topImages.sort(key=lambda x:x[1], reverse=False)
                relevant_imagenames = [x[0] for x in topImages][:t]
            
            io.display_query_results(image_dict, query_image, query_image_filename, relevant_imagenames, output_folder, 'task8_img_results.png')
            query_results_imagenames = relevant_imagenames
            
def parse_args():
    # Parsing of input arguments
    # Idea here is that we want to set up some stuff initially for the loop of user
    # input and data processing.
    
    # We will allow the following:
    #   - The user to specify LSH or VA files, and have the program construct the index in memory.
    #        Alternatively, let the program first read in a stored LHS/VA file
    #   - An input folder from which to read files (if an LSH/VA file is not specified)
    #   - An output folder to write results (may not be necessary)
    # All other parameters will be handled by the main function.
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_folder",         required=False,  help="Input image database folder to create index file")
    parser.add_argument("-f", "--index_file",           required=False,  help="Saved index file")
    parser.add_argument("-o", "--output_folder",        required=True,   help="Folder to which results are written")

    arguments = parser.parse_args()
    
    # Ensure we either read an index file or know to construct one.
    if arguments.input_folder is None and arguments.index_file is None:
        parser.error("You must specify an input folder to create an index file or a saved index file!")
    # Ensure arguments exist
    if arguments.input_folder is not None and not os.path.isdir(arguments.input_folder):
        parser.error("Input image folder does not exist!")
    if not os.path.isdir(arguments.output_folder):
        parser.error("Output folder does not exist!")
    if arguments.index_file is not None and not os.path.exists(os.path.join(os.getcwd(), arguments.index_file)):
        parser.error("Latent semantic file does not exist!")
        
    return arguments.input_folder, arguments.index_file, arguments.output_folder
    
def prep_selected_image_dict(labels, names, data):
    results = {}
    for i in range(len(labels)):
        if labels[i] == 1:
            results[names[i]] = data[i]
    return results   

def compute_vector_difference(dict, compare_vector):
    "Function to calculate distance between a vector and set of other vectors"
    output_tuple_list = []

    for img_id, features in dict.items():
        output_tuple_list.append((img_id, scipy.spatial.distance.euclidean(compare_vector, features)))
    return output_tuple_list

if __name__ == "__main__":
    main(*parse_args())
    
