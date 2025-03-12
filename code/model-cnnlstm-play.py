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

def calculate_playability_score(y_pred):
    def within_range(tensor, k):
        num_pitches = tf.cast(tf.shape(tensor)[0], dtype=tf.float32)  # 确保 num_pitches 是 float32

        max_pitch = tf.reduce_max(tf.where(tensor > 0, tf.range(num_pitches, dtype=tf.float32), -1))
        min_pitch = tf.reduce_min(tf.where(tensor > 0, tf.range(num_pitches, dtype=tf.float32), tf.cast(num_pitches, tf.float32)))

        down_start = tf.maximum(max_pitch - k, 0)
        down_end = max_pitch + 1
        down_range_mask = tf.scatter_nd(
            tf.expand_dims(tf.range(tf.cast(down_start, tf.int32), tf.cast(down_end, tf.int32)), axis=1),
            tf.ones([tf.cast(down_end - down_start, tf.int32)], dtype=tf.float32),
            [tf.cast(num_pitches, tf.int32)]
        )

        up_start = min_pitch
        up_end = tf.minimum(min_pitch + k + 1, num_pitches)
        up_range_mask = tf.scatter_nd(
            tf.expand_dims(tf.range(tf.cast(up_start, tf.int32), tf.cast(up_end, tf.int32)), axis=1),
            tf.ones([tf.cast(up_end - up_start, tf.int32)], dtype=tf.float32),
            [tf.cast(num_pitches, tf.int32)]
        )

        combined_range_mask = tf.logical_or(tf.cast(down_range_mask, dtype=tf.bool), tf.cast(up_range_mask, dtype=tf.bool))
        tensor_mask = tf.cast(tensor, dtype=tf.bool)
        out_of_range_mask = tf.logical_and(tensor_mask, tf.logical_not(combined_range_mask))

        return tf.reduce_any(out_of_range_mask)

    def calculate_hand_span_per_timestep(timestep_data):
        out_of_range_k1 = within_range(timestep_data, k=8)
        out_of_range_k2 = within_range(timestep_data, k=10)

        handspan = tf.where(tf.logical_not(out_of_range_k1), 6.0,
                            tf.where(tf.logical_not(out_of_range_k2), 10.0, 11.0))
        return handspan

    def calculate_hand_span(y_pred):
        batch_size = tf.shape(y_pred)[0]
        timesteps = tf.shape(y_pred)[1]

        y_pred_reshaped = tf.reshape(y_pred, [batch_size * timesteps, 128, 1])

        hand_spans = tf.map_fn(calculate_hand_span_per_timestep, y_pred_reshaped, dtype=tf.float32)

        hand_spans = tf.reshape(hand_spans, [batch_size, timesteps])
        max_hand_span = tf.reduce_max(hand_spans, axis=1)

        handspan_score = tf.where(max_hand_span <= 6, 0.0,
                                  tf.where(max_hand_span <= 10, 0.5, 1.0))
        return tf.reduce_mean(handspan_score)

    def calculate_num_of_notes_score(y_pred):
        batch_size = tf.shape(y_pred)[0]
        timesteps = tf.shape(y_pred)[1]

        num_of_notes = tf.reduce_sum(y_pred, axis=[2, 3])
        max_num_of_notes = tf.reduce_max(num_of_notes, axis=1)

        num_of_notes_score = tf.where(max_num_of_notes > 10, 0.0,
                                      tf.where(max_num_of_notes > 6, 0.5, 1.0))
        return tf.reduce_mean(num_of_notes_score)

    hand_span_score = calculate_hand_span(y_pred)
    num_of_notes_score = calculate_num_of_notes_score(y_pred)

    return (hand_span_score * 0.5) + (num_of_notes_score * 0.5)


def custom_loss(y_true, y_pred, lambda_param=0.1):
    input_data = y_true[..., 1:]
    y_true_only = y_true[..., :1]
    musical_loss = SelectiveBinaryCrossentropy()([input_data, y_true_only, y_pred])
    playability_score = calculate_playability_score(y_pred)
    return musical_loss + lambda_param * (1 - playability_score)

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
        total_batches = len(self.indices) // self.batch_size  # 取整數部分，避免不完整 batch
        return total_batches  # 確保 `fit()` 時不會超過 batch 界限
    
    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size : (index + 1) * self.batch_size]

        # 確保 batch 大小與 batch_size 一致
        if len(batch_indices) < self.batch_size:
            return None  # 返回 None，TensorFlow 會自動忽略這個 batch

        X_batch = self.X[batch_indices].transpose(0, 2, 3, 1)
        Y_batch = self.Y[batch_indices].transpose(0, 2, 3, 1)
        return X_batch, np.concatenate([Y_batch, X_batch], axis=-1)
    
    def on_epoch_end(self):
        np.random.shuffle(self.indices)

# 训练参数
X_train_path = "/home/wanchichang/piano-reduction/LOP_database/X_sw_train.npy"
Y_train_path = "/home/wanchichang/piano-reduction/LOP_database/Y_sw_train.npy"
output_folder = "cnn_lstm_model_play_512_sw_drop34_100"
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
    
    x = LSTM(512, activation="tanh", recurrent_activation="sigmoid", dropout=0.4, recurrent_dropout=0.3, return_sequences=True)(x)
    
    x = TimeDistributed(Dense(128, activation="sigmoid"))(x)
    outputs = Reshape((64, 128, 1))(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.005), loss=custom_loss, metrics=[custom_accuracy])
    model.summary()

# 训练模型
batch_size = 512
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

