# from util import getMeasure, getTrackNumber
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
import os

# 数据生成器类
class DataGenerator(Sequence):
    def __init__(
        self, X_path, Y_path, batch_size, validation_split=0.2, is_validation=False
    ):
        self.X_path = X_path
        self.Y_path = Y_path
        self.batch_size = batch_size
        self.X = np.load(X_path, mmap_mode="r")
        self.Y = np.load(Y_path, mmap_mode="r")
        self.indices = np.arange(self.X.shape[0])

        # Split the indices for training and validation
        split = int(np.floor(validation_split * len(self.indices)))
        if is_validation:
            self.indices = self.indices[:split]
        else:
            self.indices = self.indices[split:]

    def __len__(self):
        return int(np.floor(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        X_batch = self.X[batch_indices]
        Y_batch = self.Y[batch_indices]

        # 调整输入数据的形状
        X_batch = X_batch.transpose(
            0, 2, 3, 1
        )  # 将形状从 (batch_size, 4, 64, 128) 调整为 (batch_size, 64, 128, 4)
        Y_batch = Y_batch.transpose(
            0, 2, 3, 1
        )  # 将形状从 (batch_size, 1, 64, 128) 调整为 (batch_size, 64, 128, 1)

        return X_batch, Y_batch

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

# absolute path in ssh server
X_train_path = "/home/wanchichang/piano-reduction/LOP_database/X_train.npy"
Y_train_path = "/home/wanchichang/piano-reduction/LOP_database/Y_train.npy"

batch_size = 16
validation_split = 0.2

# from tensorflow.keras.metrics import CosineSimilarity
train_generator = DataGenerator(
    X_train_path, Y_train_path, batch_size, validation_split, is_validation=False
)
validation_generator = DataGenerator(
    X_train_path, Y_train_path, batch_size, validation_split, is_validation=True
)


# 定义模型
def simple_attention(query, value):
    key_dim = query.shape[-1]
    scores = tf.matmul(
        query, query, transpose_b=True
    )  # (batch_size, seq_length, seq_length)
    scores = scores / tf.sqrt(tf.cast(key_dim, tf.float32))
    weights = tf.nn.softmax(scores, axis=-1)  # Attention weights
    output = tf.matmul(weights, value)  # (batch_size, seq_length, value_dim)
    return output


input_shape = (64, 128, 4)  # Updated input shape

inputs = tf.keras.layers.Input(shape=input_shape)

x = tf.keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same")(inputs)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.Reshape((64, 128 * 64))(x)  # Flatten keeping the sequence dimension

x = tf.keras.layers.Permute((2, 1))(x)  # Transpose for attention
x = simple_attention(x, x)
x = tf.keras.layers.Permute((2, 1))(x)  # Transpose back

x = tf.keras.layers.LSTM(512, return_sequences=True)(x)

x = tf.keras.layers.Reshape((64, 512))(x)

# 改变输出层，使其输出形状为 (64, 128, 1)
x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(128, activation="sigmoid"))(
    x
)  # Predicting for each pitch
outputs = tf.keras.layers.Reshape((64, 128, 1))(x)  # Adjust to match the label shape

model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
# model.compile(optimizer='adam', loss='mse', metrics=[CosineSimilarity(name='cosine_similarity')])
model.summary()

history = model.fit(train_generator, epochs=10, validation_data=validation_generator)
import matplotlib.pyplot as plt

# history = model.fit(X_train, Y_train, epochs=10, batch_size=32, validation_split=0.2)

# 繪製 training loss 和 validation loss
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.xticks(np.arange(1, len(history.history["loss"]) + 1, step=2))
plt.legend()
plt.savefig("training_validation_loss.png")
plt.show()
