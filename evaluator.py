import sys
from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def main(model_path):
    datagen = ImageDataGenerator(
        samplewise_center=True,
        rotation_range=90,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True
    )

    valid_it = datagen.flow_from_directory('dataset/fruit/test', target_size=(
        224, 224), color_mode='rgb', class_mode="categorical", batch_size=4)

    model = models.load_model(model_path)
    evaluation = model.evaluate(valid_it, steps=valid_it.samples/valid_it.batch_size)
    print(evaluation)

if __name__ == '__main__':
    if len(sys.argv) < 2 and len(sys.argv) > 2:
        print('Invalid invokation - Please pass the model to be evaluated.')
        exit(0)
    else:
        model = sys.argv[1]
        try:
            main(model)
        except KeyboardInterrupt:
            exit(0)
