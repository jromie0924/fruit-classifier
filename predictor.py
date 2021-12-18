import os
import sys
import numpy as np
from tensorflow.keras import models
from tensorflow.keras.preprocessing import image as image_utils
from tensorflow.keras.applications.vgg16 import preprocess_input
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

MODEL_NAME = '_model'
CLASSES = ['Apple', 'Banana', 'Orange', 'Rotten Apple', 'Rotten Banana', 'Rotten Orange']

def show_image(path):
    image = mpimg.imread(path)
    print(image.shape)
    plt.imshow(image)
    plt.show()

def load_and_process_image(path):
    image = image_utils.load_img(path, target_size=(224,224))
    image = image_utils.img_to_array(image)
    image = image.reshape(1, 224, 224, 3)
    image = preprocess_input(image)
    return image

def predict(model, image):
    prediction = model.predict(image)[0]
    index = np.where(prediction == 1)[0][0]

    print(f'\n\n~~~~~ {CLASSES[index]} ~~~~~\n\n')

def main(path):
    model = None
    if os.path.isdir(f'./{MODEL_NAME}'):
        model = models.load_model(MODEL_NAME)
        model.summary()
        show_image(path)
        image = load_and_process_image(path)
        predict(model, image)
    else:
        raise(NotADirectoryError(f'{MODEL_NAME} does not exist.'))


if __name__ == '__main__':
    if len(sys.argv) < 2 and len(sys.argv) > 2:
        print('Must pass ONE image path to the classifier.')
        exit(0)
    try:
        path = sys.argv[1]
        main(path)
    except KeyboardInterrupt:
        exit(0)