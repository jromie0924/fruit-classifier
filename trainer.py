import os
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from logger import logger as logz

logger = logz()

def main():
    logger.log('Utilizing the VGG16 keras model')
    base_model = keras.applications.VGG16(
        weights='imagenet',
        input_shape=(224, 224, 3),
        include_top=False
    )

    # logger.log('Utilizing the VGG19 keras model')
    # base_model = keras.applications.VGG19(
    #     weights='imagenet',
    #     input_shape=(224, 224, 3),
    #     include_top=False
    # )

    # Freeze base model
    base_model.trainable = False
    inputs = keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = keras.layers.Flatten()(x)
    outputs = keras.layers.Dense(6, activation='softmax')(x)
    model = keras.Model(inputs, outputs)
    model.summary()
    model.compile(loss=keras.losses.CategoricalCrossentropy(
        from_logits=False), metrics=[keras.metrics.Accuracy()])

    # Data augmentation for better results
    datagen = ImageDataGenerator(
        samplewise_center=True,
        rotation_range=90,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True
    )

    # datagen = ImageDataGenerator(
    #     samplewise_center=True,
    #     rotation_range=150,
    #     zoom_range=0.1,
    #     width_shift_range=0.1,
    #     height_shift_range=0.1,
    #     horizontal_flip=True,
    #     vertical_flip=True
    # )

    train_it = datagen.flow_from_directory('dataset/fruit/train/',
                                        target_size=(224, 224),
                                        color_mode='rgb',
                                        class_mode="categorical",
                                        batch_size=4)

    valid_it = datagen.flow_from_directory('dataset/fruit/test',
                                        target_size=(224, 224),
                                        color_mode='rgb',
                                        class_mode="categorical",
                                        batch_size=4)

    # Assess old model if exists
    old_accuracy = 0
    if os.path.isfile('_model'):
        old_model = models.load_model('_model')
        eval = old_model.evaluate(valid_it, steps=valid_it.samples/valid_it.batch_size)
        old_accuracy = eval[1]

    logger.log('Fitting model.....')
    model.fit(train_it,
            validation_data=valid_it,
            steps_per_epoch=train_it.samples/train_it.batch_size,
            validation_steps=valid_it.samples/valid_it.batch_size,
            epochs=20)

    # Compare results
    evaluation = model.evaluate(valid_it, steps=valid_it.samples/valid_it.batch_size)
    logger.log('Original Model performance:')
    logger.log(f'LOSS: {evaluation[0]}; ACCURACY: {evaluation[1]}')
    if (old_accuracy > evaluation[1]):
        logger.log(f'ORIGINAL MODEL PERFORMS WORSE THAN THE PREVIOUS MODEL: New: {evaluation[1]} vs Old: {old_accuracy}')
    else:
        logger.log(f'ORIGINAL MODEL OUTPERFORMS THE PREVIOUS MODEL: New: {evaluation[1]} vs Old: {old_accuracy}')
    logger.log('\n\nmodel saved as \'fruit_model\'')
    model.save('fruit_model')

    logger.log('Tweaking model.....')

    # Unfreeze the base model
    base_model.trainable = True

    # Compile the model with a low learning rate
    model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=0.000001),
                loss=keras.losses.CategoricalCrossentropy(from_logits=False), metrics=[keras.metrics.Accuracy()])

    model.fit(train_it,
            validation_data=valid_it,
            steps_per_epoch=train_it.samples/train_it.batch_size,
            validation_steps=valid_it.samples/valid_it.batch_size,
            epochs=5)

    # Compare results
    evaluation = model.evaluate(valid_it, steps=valid_it.samples/valid_it.batch_size)
    logger.log('Tweaked Model performance:')
    logger.log(f'LOSS: {evaluation[0]}; ACCURACY: {evaluation[1]}')
    if (old_accuracy > evaluation[1]):
        logger.log(f'TWEAKED MODEL PERFORMS WORSE THAN THE PREVIOUS MODEL: New: {evaluation[1]} vs Old: {old_accuracy}')
    else:
        logger.log(f'TWEAKED MODEL OUTPERFORMS THE PREVIOUS MODEL: New: {evaluation[1]} vs Old: {old_accuracy}')
    model.save('fruit_model_tweaked_base')

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.err('Process stopped by a keyboard interrupt.')
        exit(0)
    logger.log('Training succeeded.')
    exit(0)