##################################################################################
# Authors: Ayush Anand, Pritam De, Sairaj Menon, Sritej Reddy,
#          Aaron Steele, Shubham Verma
# Course: CSE 515 Fall 2021, Arizona State University
# Professor: K. Selçuk Candan
# Project: Course Project Phase 3
# File: task4.py
# 
# Description: Main driver file for Task 4 of the assignment.
##################################################################################

from enum import unique
import numpy as np
import time
import os
from math import *
from argparse import ArgumentParser
from lib import feature, io, util, LSH, latent
from pympler import asizeof
import pickle
from pathlib import Path


def main(*args):
    # Start the timer and read arguments
    start_time = time.time()
    layers, hashes, t, query_image, folder_path, model, output_dir, latent_semantic_file, use_latent_bool = args
    # csv_file = model.lower() + '_face.csv'

    # Logic if the latent semantic file was NOT chosen (so we need to compute latent semantics)
    if latent_semantic_file is None:
        dim_red_technique = 'svd'
        k = 20
        image_dict, results = latent.compute_latent_semantics(folder_path, model, dim_red_technique, k, output_dir, True)
    else:
        [model, dim_red_technique, k, image_dict, results] = io.read_pickle_file(latent_semantic_file)
    orig_model = model
    
    if use_latent_bool:
        model = 'latent'
    
    # Convert the feature vectors into a data array to feed into the index. Note that here,
    # model is 'color', 'elbp', 'hog', or 'latent' to select the base vectors, or 'latent' to select the latent feature vectors.
    data_array = util.feature_dict_to_array(image_dict, model)

    random_vector = [np.random.randn(hashes, len(data_array[0])) for j in range(layers)]
    input_img = io.read_image(query_image)
    technique_result_query_image = feature.compute_single_feature(input_img, orig_model)
    # If we are using latent feature vectors, we need to compute the latent feature
    if use_latent_bool:
        technique_result_query_image = latent.map_single_image_to_latent(technique_result_query_image, dim_red_technique, results)
    
    #technique_result_query_image=image_dict[query_image][model]
    # print(technique_result_query_image)
    # print(technique_result_query_image.shape)
    topImages = util.compute_vector_difference_of_feature(image_dict,technique_result_query_image,model)
    #print(topImages[0][0])
    topImages.sort(key = lambda x: x[1])
    topKNormal = []
    for i in range(0,t):
        topKNormal.append(topImages[i][0])

    Lsh = []
    Query = []
    buckets = 0
    relevant_images = []

    # Create the LSH Structure and Query
    Lsh, Query, lsh_conversion_vector = LSH.LSH(hashes,data_array,layers,technique_result_query_image,image_dict, query_image)
    # Print the number of bytes for the structure. Make sure to use asizeof to reflect
    # the true size of all components
    print("bytes:", asizeof.asizeof(LSH))
    # Get the number of buckets and 'relevant images'
    relevant_images, buckets = LSH.get_relevant_images(Lsh, Query, layers, t)
    print("Total Buckets Visited:",buckets)

    #print(query_layered_hash_bucket)
    overall_images=len(relevant_images)
    relevant_images=list(set(relevant_images))
    unique_images=len(relevant_images)

    print("Overall Images Visited:", overall_images)
    print("Unique Images Visited:", unique_images)

    relevant_images_vector=[]
    final=[]
    for i in relevant_images:
        image_index=list(image_dict.keys()).index(i)
        relevant_images_vector.append([i,data_array[image_index]])
        final.append([i, LSH.calculate_euclidean_distance(data_array[image_index],technique_result_query_image)])
    final = sorted(final, key=lambda p: p[1])

    # LSH.create_html(final, "task5_output.html",t)

    resultant_relevant_images_vector=[]
    for i in relevant_images_vector:
        if i[0] in [item[0] for item in final[:t]]:
            resultant_relevant_images_vector.append(i)

    topKFound = [x[0] for x in final][:t]
    # Now, we print the top t images from the input query
    print(f"{t} similar images for the query image {query_image}")
    print(" ")
    for i in range(t):
        print(f"ImageId: {topKFound[i]}")
        
    # print(LSH.get_t_most_similar(Lsh, Query, technique_result_query_image, layers, t, image_dict, model))
    
    # Get the intersection of the returned results and what we would have gotten in a standard
    # query on the original data using Euclidean distance
    # TODO: Do FPR and MR
    list1_as_set = set(topKNormal)
    intersection = list1_as_set.intersection(topKFound)
    intersection_as_list = list(intersection)
    print("Intersection of results and standard Euclidean distance:")
    print(intersection_as_list)
    print("The total misses from top t:",t - len(intersection_as_list))
    print("The total false-positives from top t:",t - len(intersection_as_list))
    
    print(f"\nThe miss rate for the top t: {(t - len(intersection_as_list)) / t}")
    print(f"The false positive rate for the top t: {(t - len(intersection_as_list)) / t}")
    temp1 = unique_images - (len(intersection_as_list))
    print(f"\nIncluding all images returned from the initial query (before filtering), the false positive rate is: {temp1/unique_images}\n")
    io.display_query_results(image_dict,input_img, query_image, topKFound, output_dir, 'task4_img_results.png')
    # query_image_result=[query_image,technique_result_query_image]
    
    # Save the results from the LSH query and the index itself
    if folder_path is not None:
        lsh_out_name = f"lsh_{folder_path.replace(r'/', '-')}_{layers}_{hashes}.p"
    else:
        lsh_out_name = f"lsh_{Path(latent_semantic_file).stem}_{layers}_{hashes}.p"
    to_dump = ['lsh', layers, hashes, Lsh, lsh_conversion_vector, image_dict, model, folder_path, dim_red_technique, results]
    pickle.dump(to_dump, open(os.path.join(output_dir, lsh_out_name), "wb"))
    print(f"Saved LSH index file to '{os.path.join(output_dir, lsh_out_name)}'!")
    if folder_path is not None:
        io.save_query_results(folder_path, output_dir, input_img, query_image, technique_result_query_image, final, 4, t)
    else:
        io.save_query_results(Path(latent_semantic_file).stem, output_dir, input_img, query_image, technique_result_query_image, final, 4, t)
    print(f"Size in bytes of everything stored for LSH: {asizeof.asizeof(to_dump)}")
    print("Execution time:%s seconds"%(time.time()-start_time))

def parse_args():
    """
    Parse the input arguments for the program.
    """
    # Parsing of input arguments
    
    parser = ArgumentParser()
    parser.add_argument("-l", "--layers",           required=True,  type=int, help="Number of Layers")
    parser.add_argument("-k", "--hashes",           required=True,  type=int, help="Number of Hashes per Layer")
    parser.add_argument("-t", "--similar_image",    required=True,  type=int, help="Number of similar Images")
    parser.add_argument("-q", "--query_image",      required=True,  type=str, help="query image")
    parser.add_argument("-f", "--folder_path",      required=False,  type=str, help="folder path")
    parser.add_argument("-m", "--model",            required=True,  type=str, help="Model to compute feature vector")
    parser.add_argument("-o", "--output_directory", required=True,  type=str, help="Directory to which results are written")
    parser.add_argument("-d", "--latent_file",      required=False, type=str, help="Stored latent semantic file (not required)")
    parser.add_argument("-x", action='store_true',                            help="Flag to indicate whether or not to use latent features")
    arguments = parser.parse_args()

    # Process command-line arguments and ensure that inputs are valid
    if arguments.layers < 1:
        parser.error("The number of layers should be at least 1!")
    if arguments.hashes < 1:
        parser.error("The number of hashes per layer should be at least 1!")
    if arguments.similar_image < 1:
        parser.error("At least 1 similar image needs to be returned by the query!")
    
    if not os.path.exists(os.path.join(os.getcwd(), arguments.query_image)):
        parser.error("Query image does not exist!")
    if arguments.folder_path is not None and not os.path.isdir(arguments.folder_path):
        parser.error("Input image folder path does not exist!")
    if not os.path.isdir(arguments.output_directory):
        parser.error("Output directory does not exist!")
    
    if arguments.latent_file is not None and not os.path.exists(os.path.join(os.getcwd(), arguments.latent_file)):
        parser.error("If you include a latent semantic file, it must be valid!")
        
    if arguments.model not in io.MODEL:
        parser.error(f"Invalid model chosen! Valid options are {io.MODEL}.")

    return arguments.layers, arguments.hashes, arguments.similar_image, arguments.query_image, arguments.folder_path, arguments.model, arguments.output_directory, arguments.latent_file, arguments.x

if __name__ == "__main__":
    main(*parse_args())