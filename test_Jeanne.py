#------------------------------------------------------------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import argparse
import random
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score, matthews_corrcoef
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
from sklearn.utils.class_weight import compute_class_weight
#------------------------------------------------------------------------------------------------------------------

'''
 LAST UPDATE 10/20/21
 02/14/2022 LSDR CHECK CONSISTENCY
'''
#------------------------------------------------------------------------------------------------------------------
SEED = 123
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
weight_init = glorot_uniform(seed=SEED)
rng = tf.random.Generator.from_seed(SEED, alg='philox')

CHANNELS = 3
IMAGE_SIZE = 100

##  0, 1, 3
## Review documentation on tersorflow https://www.tensorflow.org/api_docs/python/tf/io/decode_jpeg

parser = argparse.ArgumentParser()
parser.add_argument("--path", default=None, type=str, required=True)  # Path of file
parser.add_argument("--split", default=False, type=str, required=True)  # validate, test, train

args = parser.parse_args()

PATH = args.path
DATA_DIR = args.path + os.path.sep + 'Data' + os.path.sep
SPLIT = args.split


'''PATH = '/home/ubuntu/Exam1-v4'
DATA_DIR = PATH + os.path.sep + 'Data' + os.path.sep
SPLIT = "test"'''

n_epoch = 2
BATCH_SIZE = 32
LR = 0.1
DROPOUT = 0.2

## Process images in parallel
AUTOTUNE = tf.data.AUTOTUNE

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

def augment(image):

    #g = tf.random.Generator.from_non_deterministic_state()
    # Make a new seed.
    #img_seed = tf.random.split(seed, num=1)[0, :]
    img_seed = rng.make_seeds(2)[0]

    image = tf.image.resize( image, [IMAGE_SIZE, IMAGE_SIZE])
    # Random crop back to the original size.
    image = tf.image.stateless_random_crop(image, size=(int(IMAGE_SIZE*.8),int(IMAGE_SIZE*.8),CHANNELS), seed=img_seed)
    # Random flip left right
    image = tf.image.stateless_random_flip_left_right(image, seed=img_seed)
    # Random saturation
    image = tf.image.stateless_random_saturation(image, 0.1, 4.0, seed=img_seed)
    # Random flip up down
    #image = tf.image.stateless_random_flip_up_down(image, seed=img_seed)
    return image

#------------------------------------------------------------------------------------------------------------------

def process_path(feature, target):
    '''
             feature is the path and id of the image
             target is the result
             returns the image and the target as label
    '''

    label = target

    # Processing feature

    # load the raw data from the file as a string
    file_path = feature

   ## REad the image from disk
    img = tf.io.read_file(file_path)
    img = tf.io.decode_image(img, channels=CHANNELS, expand_animations=False)

    ## Randomly augment images
    if train == True:
        img = augment(img)

    ## Resize the image
    #img = tf.image.resize_with_crop_or_pad( img, 300, 300)
    img = tf.image.resize( img, [IMAGE_SIZE, IMAGE_SIZE])

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

def read_train_data(num_classes):
    ## Only the training set
    ## xdf_dset ( data set )
    ## read the data data from the file

    global xdf_dset

    # Drop random sample of class5 rows
    #drop_rows = xdf_dset[xdf_dset['target']=='class5'].sample(frac=0.5, random_state=SEED).index
    #xdf_dset = xdf_dset.drop(drop_rows)

    max_class_count = (xdf_dset['target'].value_counts()).iloc[0]
    for name in class_names:
        num_of_name = (xdf_dset['target']==name).sum()
        num_of_augmented_imgs = max_class_count - num_of_name
        copies = np.random.randint(0,num_of_name,num_of_augmented_imgs)
        xdf_dset = pd.concat([xdf_dset,xdf_dset[xdf_dset['target']==name].iloc[copies]],axis=0).reset_index(drop=True)

    # Shuffle the Dataframe
    xdf_dset = xdf_dset.sample(frac=1).reset_index(drop=True)

    ds_inputs = np.array(DATA_DIR + xdf_dset['id'])
    ds_targets = get_target(num_classes)

    ## Make the channel as a list to make it variable
    ## Create the data set and call the function map to create the dataloader using
    ## tf.data.Dataset
    ## dataset.map
    ## map creates an iterable
    ## More information on https://www.tensorflow.org/tutorials/images/classification

    list_ds = tf.data.Dataset.from_tensor_slices((ds_inputs,ds_targets)) # creates a tensor from the image paths and targets

    final_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE)

    return final_ds

#------------------------------------------------------------------------------------------------------------------

def read_data(num_classes):
    ## Only the training set
    ## xdf_dset ( data set )
    ## read the data data from the file

    ds_inputs = np.array(DATA_DIR + xdf_dset['id'])
    ds_targets = get_target(num_classes)

    ## Make the channel as a list to make it variable
    ## Create the data set and call the function map to create the dataloader using
    ## tf.data.Dataset
    ## dataset.map
    ## map creates an iterable
    ## More information on https://www.tensorflow.org/tutorials/images/classification

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

def model_definition():
    # Define a Keras sequential model
    model = tf.keras.Sequential()

    # Define the first dense layer
    model.add(tf.keras.layers.Dense(300, activation='relu', input_shape=(INPUTS_r,)))
    #model.add(Dropout(DROPOUT, seed=SEED))
    model.add(BatchNormalization())
    model.add(tf.keras.layers.Dense(200, activation='relu'))
    #model.add(Dropout(DROPOUT, seed=SEED))
    model.add(BatchNormalization())
    model.add(tf.keras.layers.Dense(100, activation='relu'))
    model.add(Dropout(DROPOUT, seed=SEED))
    model.add(BatchNormalization())
    model.add(tf.keras.layers.Dense(100, activation='relu'))
    model.add(Dropout(DROPOUT, seed=SEED))
    model.add(BatchNormalization())
    model.add(tf.keras.layers.Dense(80, activation='relu'))
    model.add(Dropout(DROPOUT, seed=SEED))
    model.add(BatchNormalization())
    model.add(tf.keras.layers.Dense(50, activation='relu'))

    model.add(tf.keras.layers.Dense(OUTPUTS_a, activation='softmax', kernel_initializer=weight_init)) #final layer , outputs_a is the number of targets

    model.compile(optimizer=Adam(lr=LR), loss='categorical_crossentropy', metrics=['accuracy'])

    save_model(model) #print Summary
    return model

#------------------------------------------------------------------------------------------------------------------
# https://stackoverflow.com/questions/41648129/balancing-an-imbalanced-dataset-with-keras-image-generator
def balance_model(class_names):

    unique_classes = class_names
    y = xdf_data['target'].to_numpy()

    class_weights = compute_class_weight('balanced',classes=unique_classes, y=y)
    #train_class_weights = {class_id: weight for class_id, weight in zip(unique_classes, class_weights)}
    train_class_weights = dict(enumerate(class_weights))

    return train_class_weights

#------------------------------------------------------------------------------------------------------------------

def train_func(train_ds):
    '''
        train the model
    '''

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience =100)
    check_point = tf.keras.callbacks.ModelCheckpoint('model_{}.h5'.format(NICKNAME), monitor='accuracy', save_best_only=True)
    final_model = model_definition()

    final_model.fit(train_ds,  epochs=n_epoch, callbacks=[early_stop, check_point])
    #final_model.fit(train_ds,  epochs=n_epoch, callbacks=[check_point])
    #final_model.fit(train_ds,  epochs=n_epoch, callbacks=[check_point], class_weight=train_class_weights)

#------------------------------------------------------------------------------------------------------------------

def predict_func(test_ds):

    final_model = tf.keras.models.load_model('model_{}.h5'.format(NICKNAME))
    res = final_model.predict(test_ds)
    xres = [ tf.argmax(f).numpy() for f in res]
    xdf_dset['results'] = xres

    save_model(final_model)

    ## write the rsults to a results_<nickname> xlsx
    ## the function will return a probability if using softmax
    ## The answers should be the the label not the probability
    ## for more information on activation functions and the results https://www.tensorflow.org/api_docs/python/tf/keras/activations

    xdf_dset.to_excel('results_{}.xlsx'.format(NICKNAME), index=False)
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

    ## Reading the excel file from a directory
    for file in os.listdir(PATH+ os.path.sep+ "excel"):
        if file[-5:] == '.xlsx':
            FILE_NAME = PATH+os.path.sep + "excel" + os.path.sep + file

    # Reading and filtering Excel file

    xdf_data = pd.read_excel(FILE_NAME)

    ## Multiclass , verify the classes , change from strings to numbers

    class_names = process_target(1)

    INPUTS_r = IMAGE_SIZE * IMAGE_SIZE * CHANNELS
    OUTPUTS_a = len(class_names)

    ## Processing Train dataset

    train = True
    #train_class_weights = balance_model(class_names)

    xdf_dset = xdf_data[xdf_data["split"] == 'train'].copy()

    train_ds = read_train_data( OUTPUTS_a )

    train_func(train_ds)

    # Preprocessing Test dataset
    train = False
    xdf_dset = xdf_data[xdf_data["split"] == SPLIT].copy()

    test_ds= read_data(OUTPUTS_a)

    predict_func(test_ds)

    ## Metrics Function over the result of the test dataset
    list_of_metrics = ['f1_macro','acc','coh']
    list_of_agg = ['avg','sum']
    metrics_func(list_of_metrics, list_of_agg)

if __name__ == "__main__":

    main()
