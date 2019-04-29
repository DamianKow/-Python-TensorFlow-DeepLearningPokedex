import sys
import constants
from create_dataset import create_dataset_from_images, save_data_to_pickle_file, load_data_from_pickle_file, load_pokemon_json, load_all_datasets
from cnn_model import create_cnn_model
import matplotlib.pyplot as plt
import cv2
import numpy as np

def save_datasets(x_train, y_train, x_test, y_test):
    print("Creating new dataset and saving files:")
    save_data_to_pickle_file(x_train, constants.META_DATA_X_TRAIN_PATH)
    save_data_to_pickle_file(y_train, constants.META_DATA_Y_TRAIN_PATH)
    save_data_to_pickle_file(x_test, constants.META_DATA_X_TEST_PATH)
    save_data_to_pickle_file(y_test, constants.META_DATA_Y_TEST_PATH)
    print("x_train.pickle in " + constants.META_DATA_X_TRAIN_PATH)
    print("y_train.pickle in " + constants.META_DATA_Y_TRAIN_PATH)
    print("x_test.pickle in " + constants.META_DATA_X_TEST_PATH)
    print("y_test.pickle in " + constants.META_DATA_Y_TEST_PATH)

def print_data(hist):
    train_loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    train_acc = hist.history['acc']
    val_acc = hist.history['val_acc']
    xc = range(20)

    plt.figure(1, figsize=(7, 5))
    plt.plot(xc, train_loss)
    plt.plot(xc, val_loss)
    plt.xlabel('num of Epochs')
    plt.ylabel('loss')
    plt.title('train_loss vs val_loss')
    plt.grid(True)
    plt.legend(['train', 'val'])
    # print plt.style.available # use bmh, classic,ggplot for big pictures
    plt.style.use(['classic'])
    plt.show()

    plt.figure(2, figsize=(7, 5))
    plt.plot(xc, train_acc)
    plt.plot(xc, val_acc)
    plt.xlabel('num of Epochs')
    plt.ylabel('accuracy')
    plt.title('train_acc vs val_acc')
    plt.grid(True)
    plt.legend(['train', 'val'], loc=4)
    # print plt.style.available # use bmh, classic,ggplot for big pictures
    plt.style.use(['classic'])
    plt.show()

def fit_model():
    x_train, y_train, x_test, y_test = load_all_datasets()
    x_train = x_train/255
    x_test = x_test/255
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    number_of_classes = load_pokemon_json()
    input_shape = x_train.shape[1:]
    model = create_cnn_model(input_shape, len(number_of_classes))
    hist = model.fit(x_train, y_train, batch_size=2, epochs=3, verbose=1, validation_data=(x_test, y_test))
    # print_data(hist)

if(len(sys.argv) > 1):
    if sys.argv[1] == constants.CREATE_DATASET_AND_SAVE_ARGUMENT:
        if(len(sys.argv) == 2):
            x_train, y_train, x_test, y_test = create_dataset_from_images()
            save_datasets(x_train, y_train, x_test, y_test)
        else:
            print("UNKNOWN ERROR")
    if sys.argv[1] == constants.TRAIN_CNN_ARGUMENT:
        if(len(sys.argv) == 2):
            fit_model()
        else:
            print("UNKNOWN ERROR")
else:
    print("No argument was given")