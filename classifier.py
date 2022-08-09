import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras

class_names = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']  # Классы рыб


# Визуализация предсказаний
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'green'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)

def main():
    fishes_train = keras.preprocessing.image_dataset_from_directory('fishesss/2_fish',
                                                                    labels="inferred", class_names=class_names,
                                                                    validation_split=0.2, subset='training',
                                                                    image_size=(128, 128), batch_size=1000000,
                                                                    seed=123)  # для тренировки
    fishes_test = keras.preprocessing.image_dataset_from_directory('fishesss/2_fish',
                                                                   labels="inferred", class_names=class_names,
                                                                   validation_split=0.2, subset='validation',
                                                                   image_size=(128, 128), batch_size=1000000,
                                                                   seed=123)  # для проверки

    # Визуализация данных (рыбки из обучающего набора)
    plt.figure(figsize=(50, 9))
    for images, labels in fishes_train.take(1):
        plt.subplot(1, 4, 1)
        plt.imshow(images[1].numpy().astype("uint8"))
        plt.subplot(1, 4, 2)
        plt.imshow(images[2].numpy().astype("uint8"))
        plt.subplot(1, 4, 3)
        plt.imshow(images[3].numpy().astype("uint8"))
        plt.subplot(1, 4, 4)
        plt.imshow(images[4].numpy().astype("uint8"))
        plt.colorbar()
        plt.grid(False)

    plt.show() # Показываем рыбок

    # Конвертация данных к массиву Numpy
    x_train = None
    x_train_label = None

    for image, label in tfds.as_numpy(fishes_train):
        print(type(image), type(label), label, len(label))
        x_train = image
        x_train_label = label
        print('------')

    x_test = None
    x_test_label = None

    for image, label in tfds.as_numpy(fishes_test):
        print(type(image), type(label), label, len(label))
        x_test = image
        x_test_label = label
        print('------')

    # Масштабируем значения пикселей до диапазона от 0 до 1
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    # Отобразим первые 25 изображений из тренировочного набора
    plt.figure(figsize=(15, 15))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_train[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[x_train_label[i]])

    # Построение модели
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(128, 128, 3)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(8, activation=tf.nn.softmax)
    ])

    model.compile(optimizer=tf.optimizers.Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Обучение модели
    model.fit(x_train, x_train_label, batch_size=128, epochs=30)

    # Тестирование модели на тестовых данных
    predictions = model.predict(x_test)
    num_rows = 5
    num_cols = 15
    num_images = num_rows * num_cols
    plt.figure(figsize=(2 * num_cols, 2 * num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, num_cols, i + 1)
        plot_image(i, predictions, x_test_label, x_test)

    plt.show()

if __name__ == '__main__':
    main()