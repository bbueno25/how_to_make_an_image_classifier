"""
DOCSTRING
"""
import h5py
import keras.layers as layers
import keras.models as models
import keras.optimizers as optimizers
import keras.preprocessing.image as image
import numpy
import os

img_width, img_height = 150, 150
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'

# used to rescale the pixel values from [0, 255] to [0, 1] interval
datagen = image.ImageDataGenerator(rescale=1.0/255)

train_generator = datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=16,
    class_mode='binary')

validation_generator = datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='binary')

model = models.Sequential()
model.add(layers.Convolution2D(32, 3, 3, input_shape=(img_width, img_height, 3)))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Convolution2D(32, 3, 3))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Convolution2D(64, 3, 3))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(64))
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1))
model.add(layers.Activation('sigmoid'))

model.compile(
    loss='binary_crossentropy',
    optimizer='rmsprop',
    metrics=['accuracy'])

nb_epoch = 30
nb_train_samples = 2048
nb_validation_samples = 832

model.fit_generator(
    train_generator,
    samples_per_epoch=nb_train_samples,
    nb_epoch=nb_epoch,
    validation_data=validation_generator,
    nb_val_samples=nb_validation_samples)

model.save_weights('models/basic_cnn_20_epochs.h5')
# model.load_weights('models_trained/basic_cnn_20_epochs.h5')
model.evaluate_generator(validation_generator, nb_validation_samples)

train_datagen_augmented = image.ImageDataGenerator(
    rescale=1.0/255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

train_generator_augmented = train_datagen_augmented.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='binary')

model.fit_generator(
    train_generator_augmented,
    samples_per_epoch=nb_train_samples,
    nb_epoch=nb_epoch,
    validation_data=validation_generator,
    nb_val_samples=nb_validation_samples)

model.save_weights('models/augmented_30_epochs.h5')
# model.load_weights('models_trained/augmented_30_epochs.h5')
model.evaluate_generator(validation_generator, nb_validation_samples)

model_vgg = models.Sequential()
model_vgg.add(layers.ZeroPadding2D((1, 1), input_shape=(img_width, img_height, 3)))
model_vgg.add(layers.Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
model_vgg.add(layers.ZeroPadding2D((1, 1)))
model_vgg.add(layers.Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
model_vgg.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

model_vgg.add(layers.ZeroPadding2D((1, 1)))
model_vgg.add(layers.Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
model_vgg.add(layers.ZeroPadding2D((1, 1)))
model_vgg.add(layers.Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
model_vgg.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

model_vgg.add(layers.ZeroPadding2D((1, 1)))
model_vgg.add(layers.Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
model_vgg.add(layers.ZeroPadding2D((1, 1)))
model_vgg.add(layers.Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
model_vgg.add(layers.ZeroPadding2D((1, 1)))
model_vgg.add(layers.Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
model_vgg.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

model_vgg.add(layers.ZeroPadding2D((1, 1)))
model_vgg.add(layers.Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
model_vgg.add(layers.ZeroPadding2D((1, 1)))
model_vgg.add(layers.Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
model_vgg.add(layers.ZeroPadding2D((1, 1)))
model_vgg.add(layers.Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
model_vgg.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

model_vgg.add(layers.ZeroPadding2D((1, 1)))
model_vgg.add(layers.Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
model_vgg.add(layers.ZeroPadding2D((1, 1)))
model_vgg.add(layers.Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
model_vgg.add(layers.ZeroPadding2D((1, 1)))
model_vgg.add(layers.Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
model_vgg.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

f = h5py.File('models/vgg16_weights.h5')
for k in range(f.attrs['nb_layers']):
    if k >= len(model_vgg.layers) - 1:
        break
    g = f['layer_{}'.format(k)]
    weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
    layer = model_vgg.layers[k]
    network_list = ['Convolution1D', 'Convolution2D', 'Convolution3D', 'AtrousConvolution2D']
    if layer.__class__.__name__ in network_list:
        weights[0] = numpy.transpose(weights[0], (2, 3, 1, 0))
    layer.set_weights(weights)
f.close()

train_generator_bottleneck = datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode=None,
    shuffle=False)

validation_generator_bottleneck = datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode=None,
    shuffle=False)

bottleneck_features_train = model_vgg.predict_generator(
    train_generator_bottleneck, nb_train_samples)
numpy.save(open('models/bottleneck_features_train.npy', 'wb'), bottleneck_features_train)

bottleneck_features_validation = model_vgg.predict_generator(
    validation_generator_bottleneck, nb_validation_samples)
numpy.save(open('models/bottleneck_features_validation.npy', 'wb'), bottleneck_features_validation)

train_data = numpy.load(open('models/bottleneck_features_train.npy', 'rb'))
train_labels = numpy.array([0] * (nb_train_samples // 2) + [1] * (nb_train_samples // 2))

validation_data = numpy.load(open('models/bottleneck_features_validation.npy', 'rb'))
validation_labels = numpy.array(
    [0] * (nb_validation_samples // 2) + [1] * (nb_validation_samples // 2))

model_top = models.Sequential()
model_top.add(layers.Flatten(input_shape=train_data.shape[1:]))
model_top.add(layers.Dense(256, activation='relu'))
model_top.add(layers.Dropout(0.5))
model_top.add(layers.Dense(1, activation='sigmoid'))

model_top.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

nb_epoch = 40

model_top.fit(
    train_data,
    train_labels,
    nb_epoch=nb_epoch,
    batch_size=32,
    validation_data=(validation_data, validation_labels))

model_top.save_weights('models/bottleneck_40_epochs.h5')
#model_top.load_weights('models/with-bottleneck/1000-samples--100-epochs.h5')
#model_top.load_weights(
#    '/notebook/Data1/Code/keras-workshop/models/with-bottleneck/1000-samples--100-epochs.h5')
model_top.evaluate(validation_data, validation_labels)

# Fine-tuning the top layers of a a pre-trained network
model_vgg = models.Sequential()
model_vgg.add(layers.ZeroPadding2D((1, 1), input_shape=(img_width, img_height, 3)))
model_vgg.add(layers.Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
model_vgg.add(layers.ZeroPadding2D((1, 1)))
model_vgg.add(layers.Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
model_vgg.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

model_vgg.add(layers.ZeroPadding2D((1, 1)))
model_vgg.add(layers.Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
model_vgg.add(layers.ZeroPadding2D((1, 1)))
model_vgg.add(layers.Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
model_vgg.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

model_vgg.add(layers.ZeroPadding2D((1, 1)))
model_vgg.add(layers.Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
model_vgg.add(layers.ZeroPadding2D((1, 1)))
model_vgg.add(layers.Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
model_vgg.add(layers.ZeroPadding2D((1, 1)))
model_vgg.add(layers.Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
model_vgg.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

model_vgg.add(layers.ZeroPadding2D((1, 1)))
model_vgg.add(layers.Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
model_vgg.add(layers.ZeroPadding2D((1, 1)))
model_vgg.add(layers.Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
model_vgg.add(layers.ZeroPadding2D((1, 1)))
model_vgg.add(layers.Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
model_vgg.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

model_vgg.add(layers.ZeroPadding2D((1, 1)))
model_vgg.add(layers.Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
model_vgg.add(layers.ZeroPadding2D((1, 1)))
model_vgg.add(layers.Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
model_vgg.add(layers.ZeroPadding2D((1, 1)))
model_vgg.add(layers.Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
model_vgg.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

f = h5py.File('models/vgg16_weights.h5')
for k in range(f.attrs['nb_layers']):
    if k >= len(model_vgg.layers) - 1:
        break
    g = f['layer_{}'.format(k)]
    weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
    layer = model_vgg.layers[k]
    network_list = ['Convolution1D', 'Convolution2D', 'Convolution3D', 'AtrousConvolution2D']
    if layer.__class__.__name__ in network_list:
        weights[0] = numpy.transpose(weights[0], (2, 3, 1, 0))
    layer.set_weights(weights)
f.close()

top_model = models.Sequential()
top_model.add(layers.Flatten(input_shape=model_vgg.output_shape[1:]))
top_model.add(layers.Dense(256, activation='relu'))
top_model.add(layers.Dropout(0.5))
top_model.add(layers.Dense(1, activation='sigmoid'))

top_model.load_weights('models/bottleneck_40_epochs.h5')

model_vgg.add(top_model)

for layer in model_vgg.layers[:25]:
    layer.trainable = False

# compile the model with a SGD/momentum optimizer and a very slow learning rate.
model_vgg.compile(
    loss='binary_crossentropy',
    optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
    metrics=['accuracy'])

# prepare data augmentation configuration
train_datagen = image.ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=32,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=32,
    class_mode='binary')

# fine-tune the model
model_vgg.fit_generator(
    train_generator,
    samples_per_epoch=nb_train_samples,
    nb_epoch=nb_epoch,
    validation_data=validation_generator,
    nb_val_samples=nb_validation_samples)

model_vgg.save_weights('models/finetuning_20epochs_vgg.h5')
model_vgg.load_weights('models/finetuning_20epochs_vgg.h5')
model_vgg.evaluate_generator(validation_generator, nb_validation_samples)
model.evaluate_generator(validation_generator, nb_validation_samples)
model_top.evaluate(validation_data, validation_labels)
