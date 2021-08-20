'''
ClassificationEngine.py - main engine, responsible for training the classifier,
evaluating, and testing.
It also supports saving/loading model to/from disk.
Uses tensorflow/keras for DL; opencv for images manipulations; metrics are from sklearn
'''

import numpy as np
import os
import tensorflow as tf
import cv2
from numpy import random
from tensorflow import keras

from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import random
import json
import pickle

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.applications import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.initializers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from numpy.random import seed
from tensorflow.python.distribute.collective_all_reduce_strategy import CollectiveAllReduceExtended
from sklearn.metrics import *
import seaborn as sns
import logging

import imageio

seed(1)

# directories for data/model storage

MODEL_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models')
IMAGES_DIR_TRAIN = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', 'artist_dataset_train')
IMAGES_DIR_TEST = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', 'artist_dataset_test')
CLASSES_INDICES_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'train_generator')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results')

# todo: actually use logging
logging.basicConfig(filename='results/classification_report.log', encoding='utf-8', level=logging.DEBUG)


class ClassificationEngine:

    def __init__(self):

        # todo: move everything to config
        self.IMG_WIDTH = 200
        self.IMG_HEIGHT = 200

        self.STEP_SIZE_TRAIN = 0
        self.STEP_SIZE_VALID = 0
        self.INPUT_SHAPE = (224, 224, 3)

        # self.CLASS_INDICES = {'claude_monet': 0, 'frida_kahlo': 1, 'jackson_pollock': 2, 'jose_clemente_orozco': 3,
        #                       'salvador_dali': 4, 'vincent_van_goh': 5}

    # plot sample data
    @staticmethod
    def sample_data():

        plt.figure(figsize=(20, 20))
        n = 5
        fig, axes = plt.subplots(1, n, figsize=(20, 10))

        for i in range(n):
            artist_dir = random.choice(os.listdir(IMAGES_DIR_TRAIN))
            artist_dir_path = os.path.join(IMAGES_DIR_TRAIN, artist_dir)
            file = random.choice(os.listdir(artist_dir_path))
            image_path = os.path.join(artist_dir_path, file)
            print(f'image_path = {image_path}')

            image = plt.imread(image_path)
            axes[i].imshow(image)
            axes[i].set_title("Artist: " + artist_dir.replace('_', ' '))
            axes[i].axis('off')

        plt.show()

    # create and return dataset from images folder
    def create_dataset(self, img_folder):
        img_data_array = []
        class_name = []

        for dir1 in os.listdir(img_folder):
            for file in os.listdir(os.path.join(img_folder, dir1)):
                image_path = os.path.join(img_folder, dir1, file)
                image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
                # print(image_path)
                image = cv2.resize(image, (self.IMG_HEIGHT, self.IMG_WIDTH), interpolation=cv2.INTER_AREA)
                image = np.array(image)
                image = image.astype('float32')
                image /= 255
                img_data_array.append(image)
                class_name.append(dir1)
        print(f'img_data_array size = {len(img_data_array)}')
        print(f'class_name size = {len(class_name)}')
        return img_data_array, class_name

    # create data augmented batches and return data generators, train and valid
    # todo: refactor, get rid off class_names argument
    def create_augmentations_batches(self, class_names):
        batch_size = 4  # due to limited number of samples we need to go with small batches

        train_datagen = ImageDataGenerator(validation_split=0.25,
                                           rescale=1. / 255.,
                                           # rotation_range=45,
                                           # width_shift_range=0.5,
                                           # height_shift_range=0.5,
                                           shear_range=5,
                                           # zoom_range=0.7,
                                           horizontal_flip=True,
                                           vertical_flip=True,
                                           )

        train_generator = train_datagen.flow_from_directory(directory=IMAGES_DIR_TRAIN,
                                                            class_mode='categorical',
                                                            target_size=self.INPUT_SHAPE[0:2],
                                                            batch_size=batch_size,
                                                            subset="training",
                                                            shuffle=True,
                                                            classes=class_names
                                                            )

        valid_generator = train_datagen.flow_from_directory(directory=IMAGES_DIR_TRAIN,
                                                            class_mode='categorical',
                                                            target_size=self.INPUT_SHAPE[0:2],
                                                            batch_size=batch_size,
                                                            subset="validation",
                                                            shuffle=True,
                                                            classes=class_names
                                                            )

        self.STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
        self.STEP_SIZE_VALID = valid_generator.n // valid_generator.batch_size
        print("Total number of batches = ", self.STEP_SIZE_TRAIN, "and", self.STEP_SIZE_VALID)
        # print(train_generator.class_indices)

        # save classes indices for usage in prediction
        self.set_class_indices(train_generator)

        return train_datagen, train_generator, valid_generator

    # aux method to keep class indices in a file for later usage
    @staticmethod
    def set_class_indices(train_generator):
        if not os.path.exists(CLASSES_INDICES_DIR):
            os.makedirs(CLASSES_INDICES_DIR)
        indices_file_path = os.path.join(CLASSES_INDICES_DIR, "class_indices")
        indices_file = open(indices_file_path, "wb")
        pickle.dump(train_generator.class_indices, indices_file)

    # sample augmented data from train dataset
    @staticmethod
    def sample_augmented_data(class_names, train_datagen):
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))

        random_artist = random.choice(class_names)
        random_image = random.choice(os.listdir(os.path.join(IMAGES_DIR_TRAIN, random_artist)))
        random_image_file = os.path.join(IMAGES_DIR_TRAIN, random_artist, random_image)

        # Original image
        image = plt.imread(random_image_file)
        axes[0].imshow(image)
        axes[0].set_title("An original Image of " + random_artist.replace('_', ' '))
        axes[0].axis('off')

        # Transformed image
        aug_image = train_datagen.random_transform(image)
        axes[1].imshow(aug_image)
        axes[1].set_title("A transformed Image of " + random_artist.replace('_', ' '))
        axes[1].axis('off')

        plt.show()

    # obsolete
    def fit_model_naive(self):
        # extract the image array and class name
        img_data, class_name = self.create_dataset(IMAGES_DIR_TRAIN)

        target_dict = {k: v for v, k in enumerate(np.unique(class_name))}
        print(target_dict)
        target_val = [target_dict[class_name[i]] for i in range(len(class_name))]

        model = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(self.IMG_HEIGHT, self.IMG_WIDTH, 3)),
                tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(6)
            ])
        print(model.summary())
        model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(x=np.array(img_data, np.float32), y=np.array(list(map(int, target_val)), np.float32), epochs=5)

    @staticmethod
    def plot_training(history):

        acc = history['accuracy']
        val_acc = history['val_accuracy']
        loss = history['loss']
        val_loss = history['val_loss']
        epochs = range(len(acc))

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        axes[0].plot(epochs, acc, 'r-', label='Training Accuracy')
        axes[0].plot(epochs, val_acc, 'b--', label='Validation Accuracy')
        axes[0].set_title('Training and Validation Accuracy')
        axes[0].legend(loc='best')

        axes[1].plot(epochs, loss, 'r-', label='Training Loss')
        axes[1].plot(epochs, val_loss, 'b--', label='Validation Loss')
        axes[1].set_title('Training and Validation Loss')
        axes[1].legend(loc='best')

        plt.show()

    # create DL model based on ResNet50 (todo: try ResNet35)
    def create_model(self, train_generator, valid_generator):

        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=self.INPUT_SHAPE)

        for layer in base_model.layers:
            layer.trainable = True
        # Add layers at the end
        X = base_model.output
        X = Flatten()(X)

        X = Dense(512, kernel_initializer='he_uniform')(X)
        # X = Dropout(0.5)(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)

        X = Dense(16, kernel_initializer='he_uniform')(X)
        # X = Dropout(0.5)(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)

        n_classes = 6  # todo: magic number!
        output = Dense(n_classes, activation='softmax')(X)

        model = Model(inputs=base_model.input, outputs=output)

        optimizer = Adam(lr=0.0001)
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])

        n_epoch = 6

        early_stop = EarlyStopping(monitor='val_loss', patience=20, verbose=1,
                                   mode='auto', restore_best_weights=True)

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5,
                                      verbose=1, mode='auto')

        print(f'STEP_SIZE_TRAIN = {self.STEP_SIZE_TRAIN}')
        print(f'STEP_SIZE_VALID = {self.STEP_SIZE_VALID}')

        history1 = model.fit_generator(generator=train_generator, steps_per_epoch=self.STEP_SIZE_TRAIN,
                                       validation_data=valid_generator, validation_steps=self.STEP_SIZE_VALID,
                                       epochs=n_epoch,
                                       shuffle=True,
                                       verbose=1,
                                       callbacks=[reduce_lr],
                                       use_multiprocessing=False,
                                       workers=16
                                       )

        # Freeze core ResNet layers and train again
        for layer in model.layers:
            layer.trainable = False

        for layer in model.layers[:50]:
            layer.trainable = True

        optimizer = Adam(lr=0.0001)

        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])

        n_epoch = 12
        history2 = model.fit_generator(generator=train_generator, steps_per_epoch=self.STEP_SIZE_TRAIN,
                                       validation_data=valid_generator, validation_steps=self.STEP_SIZE_VALID,
                                       epochs=n_epoch,
                                       shuffle=True,
                                       verbose=1,
                                       callbacks=[reduce_lr, early_stop],
                                       use_multiprocessing=False,
                                       workers=16
                                       )

        return history1, history2, model

    @staticmethod
    def merge_history(history1, history2):

        history = {'loss': history1.history['loss'] + history2.history['loss'],
                   'accuracy': history1.history['accuracy'] + history2.history['accuracy'],
                   'val_loss': history1.history['val_loss'] + history2.history['val_loss'],
                   'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy'],
                   'lr': history1.history['lr'] + history2.history['lr']}
        return history

    # train method: generates train and validation data, creates model and shows evaluation metrics
    def train_classifier(self):

        img_data, class_name = self.create_dataset(IMAGES_DIR_TRAIN)

        target_dict = {k: v for v, k in enumerate(np.unique(class_name))}
        class_names = list(target_dict.keys())

        print(class_names)

        train_datagen, train_generator, valid_generator = self.create_augmentations_batches(class_names)

        history1, history2, model = self.create_model(train_generator, valid_generator)

        merged_history = self.merge_history(history1, history2)
        print(merged_history)

        with open('history.txt', 'w') as f:  # todo: datetime/model mark to differentiate
            for k, v in merged_history.items():
                merged_history[k] = str(v)

            f.write(json.dumps(merged_history))

        #self.plot_training(merged_history)
        score = model.evaluate_generator(train_generator, verbose=1)
        print("Prediction accuracy on train data =", score[1])

        score = model.evaluate_generator(valid_generator, verbose=1)
        print("Prediction accuracy on CV data =", score[1])

        self.show_classfication_report(model, valid_generator, class_names)

        self.save_model(model)

    def show_classfication_report(self, model, valid_generator, tick_labels):
        # loop on each generator batch and predict
        n_classes = len(tick_labels)
        y_pred, y_true = [], []
        for i in range(self.STEP_SIZE_VALID):
            (X, y) = next(valid_generator)
            y_pred.append(model.predict(X))
            y_true.append(y)

        # create a flat list for y_true and y_pred
        y_pred = [subresult for result in y_pred for subresult in result]
        y_true = [subresult for result in y_true for subresult in result]

        # update truth vector based on argmax
        y_true = np.argmax(y_true, axis=1)
        y_true = np.asarray(y_true).ravel()

        # update prediction vector based on argmax
        y_pred = np.argmax(y_pred, axis=1)
        y_pred = np.asarray(y_pred).ravel()

        # confusion matrix
        fig, ax = plt.subplots(figsize=(10, 10))
        conf_matrix = confusion_matrix(y_true, y_pred, labels=np.arange(n_classes))
        conf_matrix = conf_matrix / np.sum(conf_matrix, axis=1)
        sns.heatmap(conf_matrix, annot=True, fmt=".2f", square=True, cbar=False,
                    cmap=plt.cm.jet, xticklabels=tick_labels, yticklabels=tick_labels,
                    ax=ax)
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')
        ax.set_title('Confusion Matrix')
        plt.show()

        print('Classification Report:')
        report = classification_report(y_true, y_pred, labels=np.arange(n_classes), target_names=tick_labels)
        print(report)
        logging.info('Classification Report:')
        logging.info(report)

    # predict a random sample from train data; todo: get a folder of unseen images
    def classify_random_sample(self):
        model = self.load_model(MODEL_DIR)

        artist_dir = random.choice(os.listdir(IMAGES_DIR_TRAIN))
        artist_dir_path = os.path.join(IMAGES_DIR_TRAIN, artist_dir)
        file = random.choice(os.listdir(artist_dir_path))
        print(f'artist_dir_path = {artist_dir_path}')
        image_path = os.path.join(artist_dir_path, file)
        print(f'image_path = {image_path}')

        image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, dsize=self.INPUT_SHAPE[0:2])
        image = np.array(image)
        image = image.astype('float32')
        image /= 255.
        image = np.expand_dims(image, axis=0)

        prediction = model.predict(image)
        prediction_probability = np.amax(prediction)
        prediction_idx = np.argmax(prediction)

        class_indices = self.get_class_indices()

        labels = class_indices
        labels = dict((v, k) for k, v in labels.items())

        title = "Actual artist = {}\nPredicted artist = {}\nPrediction probability = {:.2f} %" \
            .format(artist_dir.replace('_', ' '), labels[prediction_idx].replace('_', ' '),
                    prediction_probability * 100)

        print(title)

    # aux method to save class indices
    @staticmethod
    def get_class_indices():
        indices_file_path = os.path.join(CLASSES_INDICES_DIR, "class_indices")
        indices_file = open(indices_file_path, "rb")
        class_indices = pickle.load(indices_file)
        return class_indices

    # save model to disk
    @staticmethod
    def save_model(model):
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)

        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save(MODEL_DIR)

    # load model from disk
    @staticmethod
    def load_model(model_folder):
        try:
            model = keras.models.load_model(model_folder)
        except:
            return None
        return model

    # classify all images from a test images folder one by one; todo: batch classification
    def classify_sample_batch(self, test_images_dir):

        model = self.load_model(MODEL_DIR)
        if model:
            class_indices = self.get_class_indices()

            if not os.path.exists(RESULTS_DIR):
                os.makedirs(RESULTS_DIR)

            results_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), RESULTS_DIR,
                                             "sample_batch_classification.txt")
            results_file = open(results_file_path, 'w')

            labels = class_indices
            labels = dict((v, k) for k, v in labels.items())
            true_labels = []
            predicted_labels = []

            for dir1 in os.listdir(test_images_dir):
                for file in os.listdir(os.path.join(test_images_dir, dir1)):
                    image_path = os.path.join(test_images_dir, dir1, file)
                    image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
                    image = cv2.resize(image, dsize=self.INPUT_SHAPE[0:2])
                    image = np.array(image)
                    image = image.astype('float32')
                    image /= 255.
                    image = np.expand_dims(image, axis=0)

                    prediction = model.predict(image)
                    prediction_prob = np.amax(prediction)
                    prediction_idx = np.argmax(prediction)

                    true_label = dir1.replace('_', ' ')
                    true_labels.append(true_label)

                    predicted_label = labels[prediction_idx].replace('_', ' ')
                    predicted_labels.append(predicted_label)

                    predicted_artist = str(labels[prediction_idx])

                    predicted_artist = " ".join([name.capitalize() for name in predicted_artist.replace('_', ' ').split()])
                    true_artist = " ".join([name.capitalize() for name in true_label.replace('_', ' ').split()])

                    title = f'Actual artist = {true_artist}; Predicted artist = {predicted_artist}; ' \
                            f'Probability = {prediction_prob * 100:.2f}%\n'

                    results_file.write(title)  # python will convert \n to os.linesep

                    print(title)
                results_file.write('\n')
            results_file.close()

    # the only exposed to API method - classify a single painting given by url
    def classify_painting_by_url(self, painting_url) -> tuple[str, float]:
        print(f'MODEL_DIR = {MODEL_DIR}')
        model = self.load_model(MODEL_DIR)
        if not model:
            return None, None

        web_image = imageio.imread(painting_url)
        web_image = cv2.resize(web_image, dsize=self.INPUT_SHAPE[0:2], )
        web_image = np.array(web_image)
        web_image = web_image.astype('float32')
        web_image /= 255.
        web_image = np.expand_dims(web_image, axis=0)

        prediction = model.predict(web_image)
        prediction_probability = np.amax(prediction)
        prediction_idx = np.argmax(prediction)

        class_indices = self.get_class_indices()
        labels = dict((v, k) for k, v in class_indices.items())

        artist = str(labels[prediction_idx])
        artist = " ".join([name.capitalize() for name in artist.replace('_', ' ').split()])

        logging.info(f'Predicted artist = {artist}')
        logging.info(f'Prediction confidence = {prediction_probability * 100:.2f}%')

        return artist, prediction_probability


# testing and debugging
if __name__ == '__main__':
    pass
    # sample_data()
    # sample_augmented_data(class_names, train_datagen)

    classifier = ClassificationEngine()
    # classifier.train_classifier()

    # classifier.classify_random_sample()
    classifier.classify_sample_batch(IMAGES_DIR_TEST)
