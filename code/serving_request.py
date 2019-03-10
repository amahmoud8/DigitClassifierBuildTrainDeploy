import cv2
import argparse
import json
import requests
import os

def sendRequest(image_path,print_prob=False):
    # Preprocess input image (read as grayscale)
    img = cv2.imread(image_path, 0).reshape([28, 28, 1])
    img_normalied = img / 255.0

    payload = {
        "instances": [{'input_image': img_normalied.tolist()}]
    }

    # sending post request to TensorFlow Serving server
    r = requests.post('http://localhost:8501/v1/models/my_digit_classifier:predict', json=payload)
    pred = json.loads(r.content.decode('utf-8'))

    # fetch prediction probababilities
    prediction_list = pred['predictions'][0]
    # print prediction
    if print_prob:
        for i in range(0, len(prediction_list)):
            print("\n Class " + str(i) + ": " + str(prediction_list[i]))

    # return most probable class info
    print('\n The most probable class for ' + str(os.path.basename(image_path)) + ' is ' +
          str(prediction_list.index(max(prediction_list))) + ' with probability: ' + str(max(prediction_list)))



# Input argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=False,
                help="path of the image")
ap.add_argument("-d", "--directory", required=False,
                help="path of the directory")
ap.add_argument("-p", "--probabilities", required=False,
                help="path of the directory")
args = vars(ap.parse_args())
path_to_image = args['image']
path_to_directory = args['directory']
probabilities_flag = args['probabilities']

# check whether we need to print probabilities of all classes (not just most probable class)
if probabilities_flag==None:
    print_prob_flag = False
else:
    if(probabilities_flag=='True'):
        print_prob_flag = True
    else:
        print_prob_flag = False


if path_to_image==None and path_to_directory==None:
    print("Please provide a path to an image or to a directory of images!")
else:
    # process single image OR directory of images
    if path_to_image!=None:
        # path is a file to a single image
        sendRequest(image_path=path_to_image,print_prob=print_prob_flag)
    else:
        # path is a directory of images
        input_dir = path_to_directory
        for file in os.listdir(input_dir):
            if file.endswith(".png"):
                curr_image_path = os.path.join(input_dir, file)
                # send request
                sendRequest(image_path=curr_image_path,print_prob=print_prob_flag)





