#------------------------------------------------------------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import argparse
import random
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score, matthews_corrcoef
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.optimizers.experimental import SGD
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation, RandomFlip
from sklearn.utils.class_weight import compute_class_weight
from keras import layers
import keras
import random

#------------------------------------------------------------------------------------------------------------------

'''
LAST UPDATE 10/20/2021 LSDR
last update 10/21/2021 lsdr
02/14/2022 am LSDR CHECK CONSISTENCY
02/14/2022 pm LSDR Change result for results

'''
#------------------------------------------------------------------------------------------------------------------

# Set seed and init
SEED = 98
weight_init = glorot_uniform(seed=SEED)
rng = tf.random.Generator.from_seed(SEED, alg='philox')
#random.seed(SEED)

## Process images in parallel
AUTOTUNE = tf.data.AUTOTUNE

## folder "Data" images
## folder "excel" excel file , whatever is there is the file
## get the classes from the excel file
## folder "Documents" readme file

OR_PATH = os.getcwd()
os.chdir("..") # Change to the parent directory
PATH = os.getcwd()
DATA_DIR = os.getcwd() + os.path.sep + 'Data' + os.path.sep
sep = os.path.sep
os.chdir(OR_PATH) # Come back to the folder where the code resides , all files will be left on this directory

n_epoch = 10
BATCH_SIZE = 128 #https://medium.com/geekculture/how-does-batch-size-impact-your-model-learning-2dd34d9fb1fa
LR = 0.04
'''LR = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.1,
    decay_steps=10000,
    decay_rate=0.9)'''
DROPOUT = 0.3

## Image processing
CHANNELS = 3
IMAGE_SIZE = 200

NICKNAME = 'Jeanne'
#------------------------------------------------------------------------------------------------------------------

def process_target(target_type):
    '''
        1- Multiclass  target = (1...n, text1...textn)
        2- Multilabel target = ( list(Text1, Text2, Text3 ) for each observation, separated by commas )
        3- Binary   target = (1,0)

    :return:
    '''


    class_names = np.sort(xdf_data['target'].unique())

    if target_type == 1:

        x = lambda x: tf.argmax(x == class_names).numpy()

        final_target = xdf_data['target'].apply(x)

        final_target = to_categorical(list(final_target))

        xfinal=[]
        for i in range(len(final_target)):
            joined_string = ",".join(str(int(e)) for e in  (final_target[i]))
            xfinal.append(joined_string)
        final_target = xfinal

        xdf_data['target_class'] = final_target


    if target_type == 2:
        target = np.array(xdf_data['target'].apply(lambda x: x.split(",")))

        xdepth = len(class_names)

        final_target = tf.one_hot(target, xdepth)

        xfinal = []
        if len(final_target) ==0:
            xerror = 'Could not process Multilabel'
        else:
            for i in range(len(final_target)):
                joined_string = ",".join( str(e) for e in final_target[i])
                xfinal.append(joined_string)
            final_target = xfinal

        xdf_data['target_class'] = final_target

    if target_type == 3:
        # target_class is already done
        pass

    return class_names

#------------------------------------------------------------------------------------------------------------------

def data_augmentation(image):

    #g = tf.random.Generator.from_non_deterministic_state()
    # Make a new seed.
    #img_seed = tf.random.split(seed, num=1)[0, :]
    #img_seed = rng.make_seeds(2)[0]
    
    # Central crop
    image = tf.image.central_crop(image, 0.8)    
    # Random crop
    image = tf.image.resize( image, [IMAGE_SIZE, IMAGE_SIZE])
    crop_size = int(np.random.randint(IMAGE_SIZE*.7,IMAGE_SIZE))
    image = tf.image.random_crop(image, size=(crop_size, crop_size, CHANNELS))    
    #image = tf.image.stateless_random_crop(image, size=(crop_size,crop_size,CHANNELS))#, seed=img_seed)
    # Random flip left right
    image = tf.image.random_flip_left_right(image)#, seed=img_seed)
    # Random saturation
    image = tf.image.random_saturation(image, 0.5, 2.0)#, seed=img_seed)
    # Random contrast
    image = tf.image.random_contrast(image, 0.5, 2.0)#, seed=img_seed)
    # Random brightness
    image = tf.image.random_brightness(image, 0.5)#, seed=img_seed)
    # Grayscale
    image = tf.image.rgb_to_grayscale(image)
    
    
    return image

#------------------------------------------------------------------------------------------------------------------

def resize_and_rescale(image):

    pipeline = tf.keras.Sequential([
        layers.Resizing(IMAGE_SIZE,IMAGE_SIZE),
        layers.Rescaling(scale=1./255)
        ])
    
    return pipeline(image)

#------------------------------------------------------------------------------------------------------------------

def process_path(feature, target):

    '''
        feature is the path and id of the image
        target is the result
        returns the image and the target as label
    '''

    label = target
    file_path = feature

    img = tf.io.read_file(file_path)
    img = tf.io.decode_image(img, channels=CHANNELS, expand_animations=False)

    if train == True:
        img = data_augmentation(img)

    img = resize_and_rescale(img)
    
    #plt.imshow(img)
    #plt.imshow(np.array(img, dtype=int))
    #plt.show()

   ## Reshape the image to get the right dimensions for the initial input in the model
    img = tf.reshape(img, [-1])

    return img, label

#------------------------------------------------------------------------------------------------------------------

def get_target(num_classes):
    '''
    Get the target from the dataset
    1 = multiclass
    2 = multilabel
    3 = binary
    '''

    y_target = np.array(xdf_dset['target_class'].apply(lambda x: ([int(i) for i in str(x).split(",")])))

    end = np.zeros(num_classes)
    for s1 in y_target:
        end = np.vstack([end, s1])

    y_target = np.array(end[1:])

    return y_target

#------------------------------------------------------------------------------------------------------------------

def img_generation():

    global xdf_dset

# Drop random sample of class5 rows
    #drop_rows = xdf_dset[xdf_dset['target']=='class5'].sample(frac=0.5, random_state=SEED).index
    #xdf_dset = xdf_dset.drop(drop_rows)

    # Add additional images to underrepresented classes
    max_class_count = (xdf_dset['target'].value_counts()).iloc[0]
    for name in class_names:
        num_of_name = (xdf_dset['target']==name).sum()
        num_of_augmented_imgs = max_class_count - num_of_name
        copies = np.random.randint(0,num_of_name,num_of_augmented_imgs)
        xdf_dset = pd.concat([xdf_dset,xdf_dset[xdf_dset['target']==name].iloc[copies]],axis=0).reset_index(drop=True)

    # Shuffle the Dataframe
    xdf_dset = xdf_dset.sample(frac=1, random_state=SEED).reset_index(drop=True)

    return xdf_dset

#------------------------------------------------------------------------------------------------------------------

def read_data(num_classes):
    '''
          reads the dataset and process the target
    '''

    ds_inputs = np.array(DATA_DIR + xdf_dset['id'])
    ds_targets = get_target(num_classes)

    list_ds = tf.data.Dataset.from_tensor_slices((ds_inputs,ds_targets)) # creates a tensor from the image paths and targets

    final_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE)

    return final_ds

#------------------------------------------------------------------------------------------------------------------

def save_model(model):
    '''
         receives the model and print the summary into a .txt file
    '''
    with open('summary_{}.txt'.format(NICKNAME), 'w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))


#------------------------------------------------------------------------------------------------------------------

def leaky_relu_model(inputs):

    #inputs = keras.Input(shape=(INPUTS_r))
    x = layers.Dense(300, activation="LeakyReLU")(inputs)
    x = BatchNormalization()(x)
    x = layers.Dense(200, activation="LeakyReLU")(x)
    x = BatchNormalization()(x)
    x = layers.Dense(100, activation="LeakyReLU")(x)
    x = BatchNormalization()(x)
    x = layers.Dense(100, activation="LeakyReLU")(x)
    x = BatchNormalization()(x)
    x = layers.Dense(80, activation="LeakyReLU")(x)
    x = BatchNormalization()(x)
    x = layers.Dense(50, activation="LeakyReLU")(x)
    x = BatchNormalization()(x)

    return x

#------------------------------------------------------------------------------------------------------------------

def weighted_model():
    return


#------------------------------------------------------------------------------------------------------------------

def model_definition():

    # Block 1
    inputs = keras.Input(shape=(INPUTS_r))
    x = layers.Dense(300, activation="relu")(inputs)
    x = layers.Dropout(DROPOUT, seed=SEED)(x)
    x = BatchNormalization()(x)
    x = layers.Dense(200, activation="relu")(x)
    x = layers.Dropout(DROPOUT, seed=SEED)(x)
    x = BatchNormalization()(x)
    x = layers.Dense(100, activation="relu")(x)
    x = layers.Dropout(DROPOUT, seed=SEED)(x)
    x = BatchNormalization()(x)
    x = layers.Dense(100, activation="relu")(x)
    x = layers.Dropout(DROPOUT, seed=SEED)(x)
    x = BatchNormalization()(x)
    x = layers.Dense(50, activation="relu")(x)
    x = layers.Dropout(DROPOUT, seed=SEED)(x)
    block1_output = BatchNormalization()(x)

    # Leaky relu
    leaky_relu= leaky_relu_model(inputs)
    output_block = layers.add([block1_output, leaky_relu])

    # Output block
    x = layers.Dense(50, activation="relu")(output_block)
    x = layers.Dropout(DROPOUT, seed=SEED)(x)
    x = BatchNormalization()(x)
    x = layers.Dense(50, activation="relu")(x)
    x = layers.Dropout(DROPOUT, seed=SEED)(x)
    x = BatchNormalization()(x)
    x = layers.Dense(50, activation="relu")(x)
    x = layers.Dropout(DROPOUT, seed=SEED)(x)
    x = BatchNormalization()(x)
    outputs = layers.Dense(OUTPUTS_a, activation='softmax')(x)

    model = keras.Model(inputs, outputs)

    model.compile(optimizer=Adam(learning_rate=LR), loss='categorical_crossentropy', metrics=[tf.keras.metrics.F1Score(average='macro'),'accuracy'])
    #model.compile(optimizer=SGD(learning_rate=LR, momentum=0.9, weight_decay=0.01), loss='categorical_crossentropy', metrics=[tf.keras.metrics.F1Score(average='macro'),'accuracy'])

    save_model(model) #print Summary
    return model

#------------------------------------------------------------------------------------------------------------------

#https://stackoverflow.com/questions/41648129/balancing-an-imbalanced-dataset-with-keras-image-generator
def balance_model(class_names):

    unique_classes = class_names
    y = xdf_data['target'].to_numpy()

    class_weights = compute_class_weight('balanced',classes=unique_classes, y=y)
    #train_class_weights = {class_id: weight for class_id, weight in zip(unique_classes, class_weights)}
    train_class_weights = dict(enumerate(class_weights))

    return train_class_weights

#------------------------------------------------------------------------------------------------------------------

def train_func(train_ds, test_ds):
   
    #train the model

    #early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience =100)
    check_point = tf.keras.callbacks.ModelCheckpoint('model_{}.h5'.format(NICKNAME), monitor='val_accuracy', save_best_only=True)
    final_model = model_definition()

    #final_model.fit(train_ds,  epochs=n_epoch, callbacks=[early_stop, check_point])
    final_model.fit(train_ds,  validation_data=test_ds, epochs=n_epoch, callbacks=[check_point])
    #final_model.fit(train_ds,  epochs=n_epoch, callbacks=[check_point], class_weight=train_class_weights)

#------------------------------------------------------------------------------------------------------------------

def predict_func(test_ds):
    '''
        predict fumction
    '''

    final_model = tf.keras.models.load_model('model_{}.h5'.format(NICKNAME))
    res = final_model.predict(test_ds)
    xres = [ tf.argmax(f).numpy() for f in res]
    xdf_dset['results'] = xres
    #xdf_dset.to_excel('results_{}.xlsx'.format(NICKNAME), index=False)

#------------------------------------------------------------------------------------------------------------------

def metrics_func(metrics, aggregates=[]):
    '''
    multiple functiosn of metrics to call each function
    f1, cohen, accuracy, mattews correlation
    list of metrics: f1_micro, f1_macro, f1_avg, coh, acc, mat
    list of aggregates : avg, sum
    :return:
    '''

    def f1_score_metric(y_true, y_pred, type):
        '''
            type = micro,macro,weighted,samples
        :param y_true:
        :param y_pred:
        :param average:
        :return: res
        '''
        res = f1_score(y_true, y_pred, average=type)
        print("f1_score {}".format(type), res)
        return res

    def cohen_kappa_metric(y_true, y_pred):
        res = cohen_kappa_score(y_true, y_pred)
        print("cohen_kappa_score", res)
        return res

    def accuracy_metric(y_true, y_pred):
        res = accuracy_score(y_true, y_pred)
        print("accuracy_score", res)
        return res

    def matthews_metric(y_true, y_pred):
        res = matthews_corrcoef(y_true, y_pred)
        print('mattews_coef', res)
        return res


    # For multiclass

    x = lambda x: tf.argmax(x == class_names).numpy()
    y_true = np.array(xdf_dset['target'].apply(x))
    y_pred = np.array(xdf_dset['results'])

    # End of Multiclass

    xcont = 1
    xsum = 0
    xavg = 0

    for xm in metrics:
        if xm == 'f1_micro':
            # f1 score average = micro
            xmet = f1_score_metric(y_true, y_pred, 'micro')
        elif xm == 'f1_macro':
            # f1 score average = macro
            xmet = f1_score_metric(y_true, y_pred, 'macro')
        elif xm == 'f1_weighted':
            # f1 score average =
            xmet = f1_score_metric(y_true, y_pred, 'weighted')
        elif xm == 'coh':
             # Cohen kappa
            xmet = cohen_kappa_metric(y_true, y_pred)
        elif xm == 'acc':
            # Accuracy
            xmet =accuracy_metric(y_true, y_pred)
        elif xm == 'mat':
            # Matthews
            xmet =matthews_metric(y_true, y_pred)
        else:
            xmet =print('Metric does not exist')

        xsum = xsum + xmet
        xcont = xcont + 1

    if 'sum' in aggregates:
        print('Sum of Metrics : ', xsum )
    if 'avg' in aggregates and xcont > 0:
        print('Average of Metrics : ', xsum/xcont)
    # Ask for arguments for each metric

    print("Accuracy breakdown by class:")
    print("---------------------------")
    label_accs = {}
    for label in range(10):
        label_ind = (y_true == label)
        # extract predictions for specific true label
        pred_label = y_pred[label_ind]
        labels = y_true[label_ind]
        # compute class-wise accuracy
        label_accs[label] = accuracy_score(labels, pred_label)
        #label_accs[accuracy_score(labels, pred_label)] = label 
    for key,value in label_accs.items():
        print(f"Class {key}: {value:.3f}")

#------------------------------------------------------------------------------------------------------------------

def main():
    global xdf_data, class_names, INPUTS_r, OUTPUTS_a, xdf_dset, train

    for file in os.listdir(PATH+os.path.sep + "excel"):
        if file[-5:] == '.xlsx':
            FILE_NAME = PATH + os.path.sep + "excel" + os.path.sep + file

    # Reading and filtering Excel file
    xdf_data = pd.read_excel(FILE_NAME)

    class_names= process_target(1)  # 1: Multiclass 2: Multilabel 3:Binary

    INPUTS_r = IMAGE_SIZE * IMAGE_SIZE * CHANNELS
    OUTPUTS_a = len(class_names)

    ## Processing Train dataset

    #train_class_weights = balance_model(class_names)

    train = True
    xdf_dset = xdf_data[xdf_data["split"] == 'train'].copy()

    xdf_dset = img_generation()

    train_ds = read_data( OUTPUTS_a)

    # Preprocessing Test dataset

    train = False
    xdf_dset = xdf_data[xdf_data["split"] == 'test'].copy()

    test_ds= read_data(OUTPUTS_a)

    # Train the model & run on test set
    train_func(train_ds, test_ds)
    predict_func(test_ds)

    ## Metrics Function over the result of the test dataset
    list_of_metrics = ['f1_macro','acc','coh']
    list_of_agg = ['avg']
    metrics_func(list_of_metrics, list_of_agg)
# ------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":

    main()
#------------------------------------------------------------------------------------------------------------------

