##################################################################################
# Authors: Ayush Anand, Pritam De, Sairaj Menon, Sritej Reddy,
#          Aaron Steele, Shubham Verma
# Course: CSE 515 Fall 2021, Arizona State University
# Professor: K. Selçuk Candan
# Project: Course Project Phase 3
# File: task5.py
# 
# Description: Main driver file for Task 5 of the assignment.
##################################################################################

from joblib import disk
import numpy as np
import time
import os
from math import *
from argparse import ArgumentParser
from lib import feature, io, util, latent, va
from pympler import asizeof
import pickle
import scipy.spatial
from pathlib import Path


def main(*args):
    # Start the timer and read arguments
    start_time = time.time()
    bits, t, query_image, folder_path, model, output_dir, latent_semantic_file, use_latent_bool = args

    # Logic if the latent semantic file was NOT chosen (so we need to compute latent semantics)
    if latent_semantic_file is None:
        dim_red_technique = 'svd'
        k = 20
        image_dict, results = latent.compute_latent_semantics(folder_path, model, dim_red_technique, k, output_dir, True)
    else:
        [model, dim_red_technique, k, image_dict, results] = io.read_pickle_file(latent_semantic_file)
    orig_model = model
    
    # Using our flag to use 'original' feature vectors
    if use_latent_bool:
        model = 'latent'
        
    # Convert the feature vectors into a data array to feed into the index. Note that here,
    # model is 'color', 'elbp', 'hog', or 'latent' to select the base vectors, or 'latent' to select the latent feature vectors.
    data_array = util.feature_dict_to_array(image_dict, model)
    input_img = io.read_image(query_image)
    technique_result_query_image = feature.compute_single_feature(input_img, orig_model)
    # If we are using latent feature vectors, we need to compute the latent feature
    if use_latent_bool:
        technique_result_query_image = latent.map_single_image_to_latent(technique_result_query_image, dim_red_technique, results)
    
    # Get the top t images using standard Euclidean distance
    topImages = util.compute_vector_difference_of_feature(image_dict,technique_result_query_image,model)
    topImages.sort(key = lambda x: x[1])
    topKNormal = []
    for i in range(0,t):
        topKNormal.append(topImages[i][0])

    # Create the VA File on the input data
    ff, pp, approx, bytes_size = va.index_partition(data_array, bits)

    # print(ff, pp, approx, bytes_size)
    
    print(f"Size in bytes of the generated VA File: {bytes_size}")
    
    ans, img_considered, buckets = va.VA_SSA(data_array, ff, pp, technique_result_query_image, t * 5)
    
    # TODO: Print visited buckets!
    print("Total Buckets Visited:",buckets)
    print("Overall Images Visited:", img_considered)
    
    # We get a limited set of results from the query, and we then compute the t closest images
    # from these results to the original point to get our t results.
    returned_imgnames = [list(image_dict.keys())[int(i)] for i in ans]
    distance_tuples = []
    for img in returned_imgnames:
        distance_tuples.append((img, scipy.spatial.distance.euclidean(technique_result_query_image, image_dict[img][model])))
    # Now, sort the results and get the top t
    distance_tuples.sort(key = lambda x: x[1])
    
    returned_imgnames = [x[0] for x in distance_tuples]
    topKFound = returned_imgnames[:t]

    print(f"{t} similar images for the query image {query_image}")
    print(" ")
    for i in range(t):
        print(f"ImageId: {topKFound[i]}")

    list1_as_set = set(topKNormal)
    intersection = list1_as_set.intersection(topKFound)
    intersection_as_list = list(intersection)
    print("Intersection of results and standard Euclidean distance:")
    print(intersection_as_list)
    
    print("The total misses from top t:",t - len(intersection_as_list))
    print("The total false-positives from top t:",t - len(intersection_as_list))
    
    print(f"\nThe miss rate for the top t: {(t - len(intersection_as_list)) / t}")
    print(f"The false positive rate for the top t: {(t - len(intersection_as_list)) / t}")
    temp1 = img_considered - (len(intersection_as_list))
    print(f"\nIncluding all images returned from the initial query (before filtering), the false positive rate is: {temp1/img_considered}\n")
    io.display_query_results(image_dict,input_img, query_image, topKFound, output_dir, 'task5_img_results.png')
    
    # Save the results from the LSH query and the index itself
    if folder_path is not None:
        va_out_name = f"va_{folder_path.replace(r'/', '-')}_{bits}.p"
    else:
        temp = Path(latent_semantic_file).stem
        print(temp)
        va_out_name = f"va_{temp}_{bits}.p"
        
    to_dump = ['va', bits, ff, pp, image_dict, model, folder_path, dim_red_technique, results]
    pickle.dump(to_dump, open(os.path.join(output_dir, va_out_name), "wb"))
    print(f"Saved VA index file to '{os.path.join(output_dir, va_out_name)}'!")
    if folder_path is not None:
        io.save_query_results(folder_path, output_dir, input_img, query_image, technique_result_query_image, distance_tuples, 5, t)
    else:
        io.save_query_results(Path(latent_semantic_file).stem, output_dir, input_img, query_image, technique_result_query_image, distance_tuples, 5, t)
    print(f"Size in bytes of everything stored for VA File: {asizeof.asizeof(to_dump)}")
    print("Execution time:%s seconds"%(time.time()-start_time))
    
def parse_args():
    """
    Parse the input arguments for the program.
    """
    # Parsing of input arguments
    
    parser = ArgumentParser()
    parser.add_argument("-b", "--bits",             required=True,  type=int, help="Number of bits per dimension")
    parser.add_argument("-t", "--similar_image",    required=True,  type=int, help="Number of similar Images")
    parser.add_argument("-q", "--query_image",      required=True,  type=str, help="query image")
    parser.add_argument("-f", "--folder_path",      required=False,  type=str, help="folder path")
    parser.add_argument("-m", "--model",            required=False,  type=str, help="Model to compute feature vector")
    parser.add_argument("-o", "--output_directory", required=True,  type=str, help="Directory to which results are written")
    parser.add_argument("-d", "--latent_file",      required=False, type=str, help="Stored latent semantic file (not required)")
    parser.add_argument("-x", action='store_true',                            help="Flag to indicate whether or not to use latent features")
    arguments = parser.parse_args()

    # Process command-line arguments and ensure that inputs are valid
    if arguments.bits < 1:
        parser.error("The number of layers should be at least 1!")
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

    return arguments.bits, arguments.similar_image, arguments.query_image, arguments.folder_path, arguments.model, arguments.output_directory, arguments.latent_file, arguments.x

if __name__ == "__main__":
    main(*parse_args())