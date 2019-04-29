import constants
import json
import os
import cv2
from keras.utils import np_utils
import random
import numpy as np
import pickle

IS_DEBUG_MODE = True

def log(to_log):
    if IS_DEBUG_MODE:
        print(to_log)

def create_dataset_from_images():
    pokemon_names = load_pokemon_json()
    log(pokemon_names)
    dataset = create_features_and_labels(pokemon_names)
    x_train, y_train, x_test, y_test = create_training_and_test_set_from_dataset(dataset)

    return x_train, y_train, x_test, y_test

def load_pokemon_json():
    with open(constants.META_DATA_POKEMON_JSON_PATH, 'r') as file:
        data_json = file.read()
    json_dict = json.loads(data_json)
    pokemon_names = []
    for name in json_dict['pokemon']:
        pokemon_names.append(name)
    return pokemon_names

def create_features_and_labels(pokemon_names):
    data = []
    class_count = len(pokemon_names)
    for name in pokemon_names:
        path_to_image_folder = os.path.join(constants.IMAGES_PATH, name)
        if os.path.isdir(path_to_image_folder) == True:
            # log(path_to_image_folder)
            index = pokemon_names.index(name)
            for img in os.listdir(path_to_image_folder):
                try:
                    img_array = cv2.imread(os.path.join(path_to_image_folder, img))
                    class_index = np_utils.to_categorical(index, class_count)
                    data.append([img_array, class_index])
                except Exception as e:
                    log("Can't convert image " + img + " to image_array")
        else:
            log("Can't find folder = " + str(path_to_image_folder))

    return data

def create_training_and_test_set_from_dataset(dataset):
    random.shuffle(dataset)
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    i = 0
    max_train = int(len(dataset) * 0.8)

    for features, label in dataset:
        if(i > max_train):
            x_test.append(features)
            y_test.append(label)
        else:
            x_train.append(features)
            y_train.append(label)
        i += 1

    x_train = np.array(x_train).reshape(-1, constants.IMAGE_SIZE, constants.IMAGE_SIZE, 3)
    x_test = np.array(x_test).reshape(-1, constants.IMAGE_SIZE, constants.IMAGE_SIZE, 3)
    return x_train, y_train, x_test, y_test

def save_data_to_pickle_file(data, path):
    pickle_out = open(path, "wb")
    pickle.dump(data, pickle_out)
    pickle_out.close()

def load_data_from_pickle_file(path):
    pickle_in = open(path,"rb")
    return pickle.load(pickle_in)

def load_all_datasets():
    pickle_in = open(constants.META_DATA_X_TRAIN_PATH, "rb")
    x_train = pickle.load(pickle_in)

    pickle_in = open(constants.META_DATA_Y_TRAIN_PATH, "rb")
    y_train = pickle.load(pickle_in)

    pickle_in = open(constants.META_DATA_X_TEST_PATH, "rb")
    x_test = pickle.load(pickle_in)

    pickle_in = open(constants.META_DATA_Y_TEST_PATH, "rb")
    y_test = pickle.load(pickle_in)

    return x_train, y_train, x_test, y_test