import argparse
import gc
from datetime import datetime
import pandas as pd
import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import efficientnet.tfkeras as efn
from tensorflow.keras.models import Sequential
import scipy.stats as st
import imutils
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPool2D, BatchNormalization, Input, \
    MaxPooling2D, GlobalMaxPooling2D, concatenate
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import argparse
import os

parser = argparse.ArgumentParser(description="Make Grad-CAMs for models trained on COVID-19/Pneumonia/Healthy CT-Scans")
parser.add_argument('--model', type=str, required=True, default="EfficientNetB4", help="Type of Model. "
                                                                                       "Currently supported models: "
                                                                                       "EfficientNetB0, "
                                                                                       "EfficientNetB1, "
                                                                                       "EfficientNetB2, "
                                                                                       "EfficientNetB3, "
                                                                                       "EfficientNetB4, "
                                                                                       "EfficientNetB5, "
                                                                                       "EfficientNetB6, "
                                                                                       "EfficientNetB7, ResNet50, "
                                                                                       "ResNet50V2, ResNet101V2, "
                                                                                       "ResNet152V2, "
                                                                                       "InceptionResNetV2, vgg16, "
                                                                                       "vgg19, inception_v3,"
                                                                                       "inception_resnet_v2,"
                                                                                       "DenseNet121, DenseNet169, "
                                                                                       "DenseNet201, Xception")
parser.add_argument('--output_dir', type=str, required=True, help="Specify the output directory for the figures")
parser.add_argument('--gpu', action="store_true", help="Are you using gpus or not?")
parser.add_argument('--efn', action="store_true", help="Are you using EfficientNet?")
parser.add_argument('--gpu_select', type=str, default="GPU:0", help="GPU you wish to train on")
parser.add_argument('--csv_directory', type=str,
                    default=r"/ifs/loni/faculty/dduncan/agarg/Updated Dataset/CleanedData/PreparedCSV",
                    help="Directory storing prepared CSV Training Files")
parser.add_argument('--gpu_select_num', type=int, nargs='+', default=[0, 1, 2, 3, 4, 5, 6, 7],
                    help="GPU you wish to train on")
parser.add_argument('--callbacks', type=int, default=0, help="0: Full call backs, 1: no tensorboard, 2: no callbacks")
parser.add_argument('--venv', action='store_true', help="whether you need to activate conda")
parser.add_argument('--img_shape', type=int, required=True, help="What shape should input images be resized to")
parser.add_argument('--no_top', action="store_true", help="Quick no top classification")
parser.add_argument('--specify_directory', type=str, default="/ifs/loni/faculty/dduncan/agarg/Images/", help="Enter the directory where images you want to compute GradCAMs on are stored")
parser.add_argument('--rounds', type=int, required=True, help="Specify the number of rounds to generate results for")
parser.add_argument('--depth', type=int, required=True, help="Used to Specify the number of rounds to generate results for, but adjust the starting/ending indices for paralellezing processing")
parser.add_argument('--dropout', action="store_true", help="Use Dropout?")
parser.add_argument('--dropout_value', type=float, default=0.2, help="Value of Dropout to use")
parser.add_argument('--trained_file', type=str, required=True)
args = parser.parse_args()
os.system("conda activate gpus")
gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    # Restrict TensorFlow to only use the first GPU
    try:
        print(args.gpu_select_num)
        gpu_selector_string = ''
        print("GPU Selector String: ", gpu_selector_string)
        print("tf.config.experimental.set_visible_devices([gpus{}, 'GPU')".format(args.gpu_select_num))
        print([gpus[i] for i in args.gpu_select_num])
        tf.config.experimental.set_visible_devices([gpus[i] for i in args.gpu_select_num], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        print(e)
else:
    print("No gpus found")

model_name = args.model
img_shape = args.img_shape
print("efn", args.efn)
time_started = datetime.now().strftime("%m_%d_%Y %H:%M:%S")

data_gen = ImageDataGenerator(rescale=1. / 255,
                              horizontal_flip=True,
                              # vertical_flip=False,
                              zoom_range=0.15,
                              rotation_range=360,
                              width_shift_range=0.15,
                              height_shift_range=0.15,
                              validation_split=0.15,
                              shear_range=0.15)

output_directory = os.path.join(args.output_dir, args.model)

try:
    os.mkdir(output_directory)
except:
    print(f"{output_directory} couldn't be made or already exists")

try:
    os.mkdir(os.path.join(output_directory, 'Condensed'))
except:
    print(f"{os.path.join(output_directory, 'Condensed')} couldn't be made or already exists")


"""time = [time for time in os.listdir(output_directory) if len(time.split()) > 1 ][1]
time_path = os.path.join(output_directory, time)
hdf5_path = os.path.join(time_path, 'hdf5')
trained_file = [os.path.join(hdf5_path, trained_models) for trained_models in os.listdir(hdf5_path)][0]"""

trained_file = args.trained_file

class GradCAM:
    def __init__(self, model, classIdx, layerName=None):
        # store the model, the class index used to measure the class
        # activation map, and the layer to be used when visualizing
        # the class activation map
        self.model = model
        self.classIdx = classIdx
        self.layerName = layerName

        # if the layer name is None, attempt to automatically find
        # the target output layer
        if self.layerName is None:
            self.layerName = self.find_target_layer()
        print(self.layerName)

        self.selected_layers = self.selected_layers[::-1]

    def find_target_layer(self):
        # attempt to find the final convolutional layer in the network
        # by looping over the layers of the network in reverse order
        self.selected_layer = ""
        self.selected_layers = []
        num = 0
        for layer in reversed(model.layers):
            self.selected_layer = layer.name
            self.selected_layers.append(layer.name)
            if len(layer.output_shape) == 4:
                return layer.name

                # otherwise, we could not find a 4D layer so the GradCAM
        # algorithm cannot be applied
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")

    def compute_heatmap(self, image, eps=1e-8):
        # construct our gradient model by supplying (1) the inputs
        # to our pre-trained model, (2) the output of the (presumably)
        # final 4D layer in the network, and (3) the output of the
        # softmax activations from the model
        gradModel = Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layerName).output,
                     self.model.output])

        # record operations for automatic differentiation
        with tf.GradientTape() as tape:
            # cast the image tensor to a float-32 data type, pass the
            # image through the gradient model, and grab the loss
            # associated with the specific class index
            inputs = tf.cast(image, tf.float32)
            (convOutputs, predictions) = gradModel(inputs)
            loss = predictions  # [:, self.classIdx]
            # check with self.ClassIdx

        # use automatic differentiation to compute the gradients
        grads = 1e15*tape.gradient(loss, convOutputs)

        # compute the guided gradients
        castConvOutputs = tf.cast(convOutputs > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = castConvOutputs * castGrads * grads

        # the convolution and guided gradients have a batch dimension
        # (which we don't need) so let's grab the volume itself and
        # discard the batch
        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]

        # compute the average of the gradient values, and using them
        # as weights, compute the ponderation of the filters with
        # respect to the weights
        weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)
        # print(guidedGrads.shape)

        # grab the spatial dimensions of the input image and resize
        # the output class activation map to match the input image
        # dimensions
        (w, h) = (image.shape[2], image.shape[1])
        heatmap = cv2.resize(cam.numpy(), (w, h))

        # normalize the heatmap such that all values lie in the range
        # [0, 1], scale the resulting values to the range [0, 255],
        # and then convert to an unsigned 8-bit integer
        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype("uint8")

        # return the resulting heatmap to the calling function
        return heatmap

    def overlay_heatmap(self, heatmap, image, alpha=0.5,
                        colormap=cv2.COLORMAP_VIRIDIS):
        # apply the supplied color map to the heatmap and then
        # overlay the heatmap on the input image
        heatmap = cv2.applyColorMap(heatmap, colormap)
        output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)

        # return a 2-tuple of the color mapped heatmap and the output,
        # overlaid image
        return (heatmap, output)


# Load the model architecture to use using standard weights for now.
# We do this to obtain the network architecture. We'll load weights from the trained model later
# At the time of script creation, EfficentNet wasn't included within standard keras libraries and had to be imported
# directly from PyPI
if args.gpu:
    if args.efn:
        model = eval(
            "efn.{}(weights='noisy-student', include_top=False, input_shape=(img_shape, img_shape, 3))".format(
                args.model))
    else:
        exec("from tensorflow.keras.applications import {}".format(args.model))
        model = eval(
            "{}(weights='imagenet', include_top=False, input_shape = (img_shape, img_shape, 3))".format(args.model))
else:
    if args.efn:
        model = eval(
            "efn.{}(weights='noisy-student', include_top=False, input_shape=(img_shape, img_shape, 3))".format(
                args.model))
    else:
        exec("from tensorflow.keras.applications import {}".format(args.model))
        model = eval(
            "{}(weights='imagenet', include_top=False, input_shape = (img_shape, img_shape, 3))".format(args.model))

# Add the final layers to the network. Adjust as fit depending on the final architecture you choose
print("No top")
x = model.output
x = GlobalAveragePooling2D()(x)
if args.dropout:
    x = Dropout(args.dropout_value)(x)
# x = Dropout(0.2)(x)
predictions = Dense(3, activation="softmax")(x)
model = Model(inputs=model.input, outputs=predictions)
trainable_count = np.sum([K.count_params(w) for w in model.trainable_weights])
print('Trainable params: {:,}'.format(trainable_count))

def custom_loss(y_true, y_pred):
    return tf.keras.losses.categorical_crossentropy(y_true, y_pred, label_smoothing=0.1)


model.compile(optimizer=Adam(0.0001), loss=custom_loss, metrics=['accuracy'])
# Load the weights from the trained model file
exec("model.load_weights('{}')".format(trained_file))


# Function used for creating absolute file paths from relative paths.
def abs_paths(my_directory):
    file_paths = []

    for folder, subs, files in os.walk(my_directory):
        for filename in files:
            file_paths.append(os.path.abspath(os.path.join(folder, filename)))

    return file_paths


# In[6]:

# Create the heatmaps, overlay them on original images, and return the stack of images
images = abs_paths(args.specify_directory)
print(output_directory)
try:
    os.mkdir(os.path.join(output_directory, 'images'))
except:
    print("error making images directory")
# os.system("mkdir {}/images".format(output_directory))
for image in images:
    image_end = image.split('/')[-1]
    # trained_file_end = args.trained_file.split('/')[-1]
    trained_file_end = trained_file.split('/')[-1]
    orig = cv2.imread(image)
    resized = cv2.resize(orig, (img_shape, img_shape))

    image = load_img(image, target_size=(img_shape, img_shape))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255

    preds = model.predict(image)
    # print(preds)
    i = np.argmax(preds[0])
    # print(i)

    cam = GradCAM(model, i)
    heatmap = cam.compute_heatmap(image)

    heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
    (heatmap, output) = cam.overlay_heatmap(heatmap, orig, alpha=0.5)

    # output = np.hstack([orig, heatmap, output])
    output = np.hstack([orig, output])
    output = imutils.resize(output, height=700)
    # cv2.imshow("Output", output)
    # cv2.waitKey(0)
    heatmap_output = imutils.resize(heatmap, height=700)


    plt.imsave(
        fname="{}/images/{}-{}-{}-{}.png".format(output_directory, args.model, "no_top", image_end, trained_file_end),
        arr=output,
        cmap=plt.cm.jet)


    """plt.imsave(fname="{}/images/{}-{}-{}-{}.png".format(output_directory, args.model, "no_top_heatmap", image_end,
                                                        trained_file_end), arr=heatmap_output,
               cmap=plt.cm.jet)
"""