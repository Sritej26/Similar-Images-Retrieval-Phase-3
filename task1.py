##################################################################################
# Authors: Ayush Anand, Pritam De, Sairaj Menon, Sritej Reddy,
#          Aaron Steele, Shubham Verma
# Course: CSE 515 Fall 2021, Arizona State University
# Professor: K. Selçuk Candan
# Project: Course Project Phase 3
# File: task1.py
# 
# Description: Main driver file for Task 1 of the assignment.
##################################################################################

import argparse
import numpy as np
import os
from lib import io, latent, ppr, stats, util
import pandas as pd
from lib.feature import compute_features
from sklearn.preprocessing import LabelEncoder
from lib import svmAlgo,svmMultiClass
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

# Decision Tree Imports
from lib.dt import DTClassifier
from sklearn.tree import DecisionTreeClassifier

# Option to ensure full arrays are printed to the terminal.
np.set_printoptions(threshold=np.inf, suppress=True)

def main(*args):
    """
    Main function for Task 1 of Phase 3 of the CSE 515 Fall 2021 project.
    """
    input_folder, model, dim_red_technique, orig_k, output_folder, latent_semantic_file, classify_folder, classifier = args
    # io.print_start_status(1, args)
    
    if orig_k == '*':
        k = 50
    else:
        k = int(orig_k)
    
    # Logic if the latent semantic file was NOT chosen (so we need to compute latent semantics)
    if latent_semantic_file is None:
        image_dict, results = latent.compute_latent_semantics(input_folder, model, dim_red_technique, k, output_folder, True)
    else:
        [model, dim_red_technique, k, image_dict, results] = io.read_pickle_file(latent_semantic_file)

    # TODO: Some stuff here is redundant, I think we recompute various statistics on the data read in
    # to classify. Clean up later.
    train_features=results[0]
    labels=[each.split("-")[1] for each in image_dict.keys()]
    potential_labels = sorted(set(labels))
    l1=LabelEncoder()
    label_encoded_values=l1.fit_transform(labels)
    
    data_x = np.array(train_features)
    data_y=np.array(label_encoded_values)
    #print(data_y)
    image_dict_unlabeled, results_unlabeled = latent.compute_latent_semantics(classify_folder, model, dim_red_technique, k, output_folder,False)
    
    test_features=results_unlabeled[0]
    expected_labels = [each.split("-")[1] for each in image_dict_unlabeled.keys()]
    l2=LabelEncoder()
    label_encoded_test=l2.fit_transform(expected_labels)

    data_x_test = np.array(test_features)
    data_y_test = np.array(label_encoded_test)
    

    # Once we have our latent semantics, we need to map the images in our original database to the latent space
    # We do this by adding an additional entry to our image_dict (called 'latent')
    # To access this vector, use image_dict[image_name]['latent']
    # NOTE: This is already done in the creation of the image dictionary!

    # We also need to read in all images in the second folder and map these images to the latent feature space
    comparison_image_dict = io.read_images(classify_folder)
    compute_features(comparison_image_dict, model)
    latent.map_image_dict_to_latent(comparison_image_dict, model, dim_red_technique, results)
    
    # Now, we run the classifier algorithms
    if classifier == 'svm':
        target_classes = [str(each) for each in list(np.unique(labels))]
        W=svmMultiClass.fit(data_x,pd.DataFrame(data_y))
        
        y_predicted,_ = svmMultiClass.predict(data_x_test,W)      
        
        report=classification_report(data_y_test,y_predicted,zero_division=1)
        
        file = open(classifier+"_"+"Task1.txt","w")
        file.write("Classification Report = "+"\n"+report)
        classified_tuples = util.convert_svm_results(comparison_image_dict, y_predicted, potential_labels)
    elif classifier == 'dt':
        target_classes = [str(each) for each in list(np.unique(label_encoded_values))]
        
        ## USING SKLEARN LIBRARY
        # dt_clf = DecisionTreeClassifier()
        # dt_clf = dt_clf.fit(data_x, data_y)
        # label_results = dt_clf.predict(data_x_test)

        ## USING CUSTOM DT IMPLEMENTATION
        dt_clf = DTClassifier(min_samples_split=2, max_depth=5)
        labels_arr = np.array([[el] for el in data_y]) # Converting labels to an array for the classifier function
        dt_clf.fit(data_x, labels_arr)
        label_results = np.array(dt_clf.predict(data_x_test)).astype(int)
        classified_tuples = util.convert_dt_results(comparison_image_dict, label_results, potential_labels)
    
    elif classifier == 'ppr':
        classified_tuples, potential_labels = ppr.PPR_Classifier(image_dict, comparison_image_dict, 'type', 0.85, dim_red_technique, top_n=len(image_dict)//25)

    # Compute and print false positive/miss rates, then save results to a file
    accuracy_stats, overall_accuracy = stats.compute_statistics(classified_tuples, comparison_image_dict, potential_labels, 'type')
    io.save_classification_results(output_folder, f"1_{classify_folder}_{classifier}.txt", classified_tuples, accuracy_stats, overall_accuracy)

def parse_args():
    """
    Parse the input arguments for the program.
    """
    # Parsing of input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_folder",         required=False,  help="Input image database folder")
    parser.add_argument("-m", "--model",                required=False,  help="Model to compute feature vector")
    parser.add_argument("-d", "--dim_red_technique",    required=False,  help="Technique used to reduce the number of dimensions")
    parser.add_argument("-k", "--k",                    required=False,  help="Number of latent semantics to select")
    parser.add_argument("-o", "--output_folder",        required=True,   help="Output folder to save results")
    parser.add_argument("-l", "--latent_semantic_file", required=False,  help="Latent semantic file to use")
    parser.add_argument("-f", "--classify_folder",      required=True,   help="Folder of images to classify")
    parser.add_argument("-c", "--classifier",           required=True,   help="Classification method to use")

    arguments = parser.parse_args()

    # If a latent semantic file is specified, we cannot use certain options
    if arguments.latent_semantic_file is not None:
        # Ensure we cannot choose invalid options
        if arguments.model is not None:
            parser.error("Cannot specify a model when using a latent semantic file!")
        elif arguments.dim_red_technique is not None:
            parser.error("Cannot specify a dimensionality reduction technique when using a latent semantic file!")
        elif arguments.k is not None:
            parser.error("Cannot specify a value of 'k' when using a latent semantic file!")
        elif arguments.input_folder is not None:
            parser.error("Cannot specify an input image folder when using a latent semantic file! Images are stored in the latent semantic file!")
        # Check if latent semantic file exists, if chosen
        if not os.path.exists(os.path.join(os.getcwd(), arguments.latent_semantic_file)):
            parser.error("Latent semantic file does not exist!")
    else:
        # Handle the case when we need to compute latent semantics first
        if arguments.model is None:
            parser.error("You must specify a model!")
        elif arguments.dim_red_technique is None:
            parser.error("You must specify a dimensionality reduction technique!")
        elif arguments.k is None:
            parser.error("You must specify a value of 'k'!")
        elif not arguments.k.isdigit() and arguments.k != "*":
            parser.error("Specify an integer value of k or '*'")
        elif arguments.input_folder is None:
            parser.error("You must specify an input image folder!")
        # If chosen, ensure that choices are valid
        if arguments.model not in io.MODEL:
            parser.error(f"Invalid model chosen! Valid options are {io.MODEL}.")
        if arguments.dim_red_technique not in io.DIMRED:
            parser.error(f"Invalid dimensionality reduction technique chosen! Valid options are {io.DIMRED}.")
        if arguments.k.isdigit() and int(arguments.k) <= 0:
            parser.error(f"Too few latent semantics chosen! Please ensure 'k' is at least 1.")
    # Check classifier for validity
    if arguments.classifier not in io.CLASSIFIER:
        parser.error(f"Invalid classifier chosen! Valid options are {io.CLASSIFIER}.")
    # Check if input folders/files exist
    if arguments.input_folder is not None and not os.path.isdir(arguments.input_folder):
        parser.error("Input image folder does not exist!")
    if not os.path.isdir(arguments.output_folder):
        parser.error("Output image folder does not exist!")
    if not os.path.isdir(arguments.classify_folder):
        parser.error("Folder of images to classify does not exist!")
        
    if arguments.k is not None and arguments.k == '*':
        k = '*'
    elif arguments.k is not None:
        k = int(arguments.k)
    else:
        k = None

    return arguments.input_folder, arguments.model, arguments.dim_red_technique, k, arguments.output_folder, arguments.latent_semantic_file, arguments.classify_folder, arguments.classifier

if __name__ == "__main__":
    main(*parse_args())
