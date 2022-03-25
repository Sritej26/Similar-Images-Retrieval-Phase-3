##################################################################################
# Authors: Ayush Anand, Pritam De, Sairaj Menon, Sritej Reddy,
#          Aaron Steele, Shubham Verma
# Course: CSE 515 Fall 2021, Arizona State University
# Professor: K. Selçuk Candan
# Project: Course Project Phase 3
# File: task3.py
# 
# Description: Main driver file for Task 3 of the assignment.
##################################################################################

import argparse
import numpy as np
import os
from lib import io, latent, ppr, stats, util, svmAlgo
from lib.feature import compute_features
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,classification_report
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# Decision Tree Imports
from lib.dt import DTClassifier
from sklearn.tree import DecisionTreeClassifier

# Decision Tree Imports
from lib.dt import DTClassifier
from sklearn.tree import DecisionTreeClassifier

# Option to ensure full arrays are printed to the terminal.
np.set_printoptions(threshold=np.inf, suppress=True)

def main(*args):
    """
    Main function for Task 3 of Phase 3 of the CSE 515 Fall 2021 project.
    """
    input_folder, model, dim_red_technique, orig_k, output_folder, latent_semantic_file, classify_folder, classifier = args

    if orig_k == '*':
        k = 50
    else:
        k = orig_k

    # Logic if the latent semantic file was NOT chosen (so we need to compute latent semantics)
    if latent_semantic_file is None:
        image_dict, results = latent.compute_latent_semantics(input_folder, model, dim_red_technique, k, output_folder,training=True)
    else:
        [model, dim_red_technique, k, image_dict, results] = io.read_pickle_file(latent_semantic_file)

    train_features=results[0]
    labels=[int(each.split("-")[3][:len(each.split("-")[3])-4]) for each in image_dict.keys()]
    potential_labels = sorted(set(labels))
    
    data_x = np.array(train_features)
    data_y=np.array(labels)
    

    image_dict_unlabeled, results_unlabeled = latent.compute_latent_semantics(classify_folder, model, dim_red_technique, k, output_folder,False)
    
    test_features=results_unlabeled[0]
    expected_labels = [int(each.split("-")[3][:len(each.split("-")[3])-4]) for each in image_dict_unlabeled.keys()]
    
    data_x_test=np.array(test_features)
    data_y_test=np.array(expected_labels)

    # Once we have our latent semantics, we need to map the images in our original database to the latent space
    # We do this by adding an additional entry to our image_dict (called 'latent')
    # To access this vector, use image_dict[image_name]['latent']
    # NOTE: This is already done in the creation of the image dictionary!

    # We also need to read in all images in the second folder and map these images to the latent feature space
    comparison_image_dict = io.read_images(classify_folder)
    compute_features(comparison_image_dict, model)
    latent.map_image_dict_to_latent(comparison_image_dict, model, dim_red_technique, results)

    train_features=results[0]

    image_dict_unlabeled, results_unlabeled = latent.compute_latent_semantics(classify_folder, model, dim_red_technique, k, output_folder,False)
    test_features=results_unlabeled[0]

    # Now, we run the classifier algorithms
    if classifier == 'svm':
        m=len(data_x)
        c=len(np.unique(data_y))
        print((c))
        target_classes = [str(each) for each in list(np.unique(data_y))]
        print(len(target_classes))
        print(len(np.unique(data_y_test)))
        svm=svmAlgo.svm_model_torch(m,c)
        C=[0.001,.01,.1,1,2,3,4]
        kernels=[svmAlgo.poly(1),svmAlgo.poly(2),svmAlgo.poly(3),svmAlgo.poly(4),svmAlgo.rbf(1),svmAlgo.rbf(2),
                svmAlgo.rbf(3),svmAlgo.rbf(4),svmAlgo.grpf(1,2),svmAlgo.grpf(2,3)]
        kernels_list=['svmAlgo.poly(1)','svmAlgo.poly(2)','svmAlgo.poly(3)','svmAlgo.poly(4)','svmAlgo.rbf(1)','svmAlgo.rbf(2)',
                'svmAlgo.rbf(3)','svmAlgo.rbf(4)','svmAlgo.grpf(1,2)','svmAlgo.grpf(2,3)']
        
      
        svm.fit(data_x,data_y,C=0.01,iterations=3,kernel=svmAlgo.poly(1))              
        print("Training Completed")
        print("Testing Started")
        predicted_labels=svm.predict(data_x_test)
        print((predicted_labels))
        print((data_y_test))
        print(accuracy_score(data_y_test,predicted_labels))
        report=classification_report(data_y_test,predicted_labels,zero_division=1)
        print(report)
        file = open(classifier+"_"+"Task3.txt","w")
        file.write("Classification Report = "+"\n"+report)
        classified_tuples = util.convert_svm_results(comparison_image_dict, predicted_labels, potential_labels)
    elif classifier == 'dt':
        expected_labels_with_type = [each.split("-")[3] for each in image_dict_unlabeled.keys()] 
        expected_labels = [int(each.split(".")[0]) for each in expected_labels_with_type]
    
        data_x_test=np.array(test_features)
        data_y_test=np.array(expected_labels)

        data_x_dt = np.array(train_features)
        labels_dt_with_type=[each.split("-")[3] for each in image_dict.keys()]
        labels_dt=[int(each.split(".")[0]) for each in labels_dt_with_type]

        potential_labels = sorted(set(labels_dt))
        l1_dt=LabelEncoder()
        label_encoded_values_dt=l1_dt.fit_transform(labels_dt)
        data_y_dt=np.array(label_encoded_values_dt)
        
        ## USING SKLEARN LIBRARY
        # dt_clf = DecisionTreeClassifier()
        # dt_clf = dt_clf.fit(data_x_dt, data_y_dt)
        # label_results = dt_clf.predict(data_x_test)

        ## USING CUSTOM DT IMPLEMENTATION
        dt_clf = DTClassifier(min_samples_split=2, max_depth=5)
        labels_arr = np.array([[el] for el in data_y_dt]) # Converting labels to an array for the classifier function
        dt_clf.fit(data_x_dt, labels_arr)
        label_results = np.array(dt_clf.predict(data_x_test)).astype(int)

        print(data_y_test, "EXPECTED")
        print(label_results, "ACTUAL")

        classified_tuples = util.convert_dt_results(comparison_image_dict, label_results, potential_labels)
    elif classifier == 'ppr':
        classified_tuples, potential_labels = ppr.PPR_Classifier(image_dict, comparison_image_dict, 'id', 0.6, dim_red_technique, top_n=len(image_dict)//20)

    # TODO: Compute and print false positive/miss rates
    accuracy_stats, overall_accuracy = stats.compute_statistics(classified_tuples, comparison_image_dict, potential_labels, 'id')
    # TODO: Print/save results and labels to some file
    io.save_classification_results(output_folder, f"3_{classify_folder}_{classifier}.txt", classified_tuples, accuracy_stats, overall_accuracy)

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
