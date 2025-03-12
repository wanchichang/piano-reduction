import tensorflow as tf
from tensorflow.keras.layers import (
    Layer,
    Input,
    Conv2D,
    BatchNormalization,
    Reshape,
    LSTM,
    Dense,
    TimeDistributed,
)
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import Sequence
import numpy as np
import matplotlib.pyplot as plt
import os

# 自定义损失函数
class SelectiveBinaryCrossentropy(Layer):
    def call(self, inputs):
        input_data, y_true, y_pred = inputs
        mask = tf.reduce_sum(input_data, axis=-1) > 0
        mask = tf.expand_dims(mask, axis=-1)
        y_true_filtered = tf.boolean_mask(y_true, mask)
        y_pred_filtered = tf.boolean_mask(y_pred, mask)
        loss = tf.keras.losses.binary_crossentropy(y_true_filtered, y_pred_filtered)
        return tf.reduce_mean(loss)

def custom_loss(y_true, y_pred):
    input_data = y_true[..., 1:]
    y_true_only = y_true[..., :1]
    return SelectiveBinaryCrossentropy()([input_data, y_true_only, y_pred])

# 自定义准确率函数
class SelectiveAccuracy(Layer):
    def call(self, inputs):
        input_data, y_true, y_pred = inputs
        mask = tf.reduce_sum(input_data, axis=-1) > 0
        mask = tf.expand_dims(mask, axis=-1)
        y_true_filtered = tf.boolean_mask(y_true, mask)
        y_pred_filtered = tf.boolean_mask(y_pred, mask)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(y_true_filtered, tf.round(y_pred_filtered)), tf.float32))
        return accuracy

def custom_accuracy(y_true, y_pred):
    input_data = y_true[..., 1:]
    y_true_only = y_true[..., :1]
    return SelectiveAccuracy()([input_data, y_true_only, y_pred])

# 数据生成器
class DataGenerator(Sequence):
    def __init__(self, X_path, Y_path, batch_size, validation_split=0.2, is_validation=False):
        self.X = np.load(X_path, mmap_mode="r")
        self.Y = np.load(Y_path, mmap_mode="r")
        self.batch_size = batch_size
        self.indices = np.arange(self.X.shape[0])
        split = int(np.floor(validation_split * len(self.indices)))
        self.indices = self.indices[:split] if is_validation else self.indices[split:]
    
    def __len__(self):
        return int(np.floor(len(self.indices) / self.batch_size))
    
    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        X_batch = self.X[batch_indices].transpose(0, 2, 3, 1)
        Y_batch = self.Y[batch_indices].transpose(0, 2, 3, 1)
        return X_batch, np.concatenate([Y_batch, X_batch], axis=-1)
    
    def on_epoch_end(self):
        np.random.shuffle(self.indices)

# 训练参数
X_train_path = "/home/wanchichang/piano-reduction/LOP_database/X_train.npy"
Y_train_path = "/home/wanchichang/piano-reduction/LOP_database/Y_train.npy"
output_folder = "cnn_lstm_model_256"
os.makedirs(output_folder, exist_ok=True)

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    input_shape = (64, 128, 4)
    inputs = Input(shape=input_shape)
    
    x = Conv2D(16, (3, 3), activation="relu", padding="same", kernel_regularizer=regularizers.l2(0.01))(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), activation="relu", padding="same", kernel_regularizer=regularizers.l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation="relu", padding="same", kernel_regularizer=regularizers.l2(0.01))(x)
    x = BatchNormalization()(x)
    
    x = Reshape((64, 128 * 64))(x)
    
    x = LSTM(256, activation="tanh", recurrent_activation="sigmoid", dropout=0.4, recurrent_dropout=0.3, return_sequences=True)(x)
    
    x = TimeDistributed(Dense(128, activation="sigmoid"))(x)
    outputs = Reshape((64, 128, 1))(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.005), loss=custom_loss, metrics=[custom_accuracy])
    model.summary()

# 训练模型
batch_size = 256
epochs = 100
train_generator = DataGenerator(X_train_path, Y_train_path, batch_size, is_validation=False)
validation_generator = DataGenerator(X_train_path, Y_train_path, batch_size, is_validation=True)

# early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint(filepath=f"{output_folder}/best_model.keras", monitor="val_loss", save_best_only=True)

history = model.fit(train_generator, epochs=epochs, validation_data=validation_generator, callbacks=[checkpoint])
model.save(f"{output_folder}/model.keras")

model.save(f"{output_folder}/model.h5")

# 画图
num_epochs = len(history.history["loss"])
x_ticks_interval = num_epochs // 10 if num_epochs >= 10 else 1
epochs_range = np.arange(1, num_epochs + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs_range, history.history["loss"], label="Training Loss")
plt.plot(epochs_range, history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.xticks(np.arange(10, num_epochs + 1, step=x_ticks_interval))
plt.legend()
plt.savefig(f"{output_folder}/training_validation_loss.png")
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(epochs_range, history.history["custom_accuracy"], label="Training Accuracy")
plt.plot(epochs_range, history.history["val_custom_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.xticks(np.arange(10, num_epochs + 1, step=x_ticks_interval))
plt.legend()
plt.savefig(f"{output_folder}/training_validation_accuracy.png")
plt.close()

