import tensorflow as tf

import tensorflow as tf
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from sklearn.model_selection import train_test_split


#объект для считывания файла, header none - нумерация столбцов
train_df = pd.read_csv('C:/COVID dataset/train.txt', sep=" ", header=None)
#Создание колонок
train_df.columns = ['patient id', 'file_paths', 'labels', 'data source']
#проход по столбцам, axis=1, удаляет указанные метки
train_df = train_df.drop(['patient id', 'data source'], axis=1)

train_df.head()

test_df = pd.read_csv('C:/COVID dataset/test.txt', sep=" ", header=None)
test_df.columns = ['id', 'file_paths', 'labels', 'data source']
test_df = test_df.drop(['id', 'data source'], axis=1)

test_df.head()

train_path = 'C:/COVID dataset/train/'
test_path = 'C:/COVID dataset/test/'

#вывод позитивных и негативных меток
train_df['labels'].value_counts()

file_count = 5000
samples = []
#массив из двух элементов positive/negative
#делает срез из двух категорий по 5000
for category in train_df['labels'].unique():
    #делаем срез из элементов, где labels == @category
    category_slice = train_df.query("labels == @category")
    #выбирает случайно строку 5000 раз без повторений
    samples.append(category_slice.sample(file_count, replace=False, random_state=1))
    #слияние в одну базу
train_df = pd.concat(samples, axis=0).sample(frac=1.0, random_state=1).reset_index(drop=True)
print(train_df['labels'].value_counts())
print(len(train_df))

#Делит в пропорции 1 к 9 (90% на обучение, 10% на проверку)
#обучающая, тестовая(готовая нейронка) и валидационная
train_df, valid_df = train_test_split(train_df, train_size=0.9, random_state=0)

print(train_df.labels.value_counts())
print(valid_df.labels.value_counts())
print(test_df.labels.value_counts())


target_size = (224, 224)
#кол-во объектов для предугадания результатов между эпохами
batch_size = 64

#Подготовка изображений для обучения
train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.resnet_v2.preprocess_input,
    #аугментация данных(увиличение объёма данных)
    horizontal_flip=True, zoom_range=0.1
)

test_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.resnet_v2.preprocess_input
)

#подгоовка изображений для обучения
train_gen = train_datagen.flow_from_dataframe(
    train_df,
    directory=train_path,
    x_col='file_paths',
    y_col='labels',
    target_size=target_size,
    batch_size=batch_size,
    color_mode='rgb',
    class_mode='binary'
)


valid_gen = test_datagen.flow_from_dataframe(
    valid_df,
    directory=train_path,
    x_col='file_paths',
    y_col='labels',
    target_size=target_size,
    batch_size=batch_size,
    color_mode='rgb',
    class_mode='binary'
)

test_gen = test_datagen.flow_from_dataframe(
    test_df,
    directory=test_path,
    x_col='file_paths',
    y_col='labels',
    target_size=target_size,
    batch_size=batch_size,
    color_mode='rgb',
    class_mode='binary'
)

#Отключение полносвязного уровня, оставлнеие свёрточного слоя
base_model = tf.keras.applications.ResNet50V2(include_top=False, input_shape=(224, 224, 3))

#последовательная модель
model = tf.keras.Sequential([
    base_model,
    #слой пуллинга
    tf.keras.layers.GlobalAveragePooling2D(),
    #полносвязный слой, 128 нейронов
    tf.keras.layers.Dense(128, activation='relu'),
    #слой батч-нормализации
    tf.keras.layers.BatchNormalization(),
    #чтобы не было переобучения
    tf.keras.layers.Dropout(0.2),
    #Выходной слой
    tf.keras.layers.Dense(1, activation='sigmoid')
])

#компиляция модели перед обучением
#скорость обучения
lr = 0.001
#функция потери, оптимизатор, метрики
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=lr), metrics=['accuracy', 'Precision', 'Recall', 'AUC', 'TruePositives', 'TrueNegatives', 'FalsePositives', 'FalseNegatives'])
#
patience = 1
stop_patience = 3
factor = 0.5

#Колбэки для оптимизации обучения
callbacks = [
    tf.keras.callbacks.ModelCheckpoint("classify_model.h5", save_best_only=True, verbose=0),
    tf.keras.callbacks.EarlyStopping(patience=stop_patience, monitor='val_loss', verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=factor, patience=patience, verbose=1)
]
epochs = 3
#verbose - вывод информации об обучении
history = model.fit(train_gen, validation_data=valid_gen, epochs=epochs, callbacks=callbacks, verbose=1)
#history = model.history()
