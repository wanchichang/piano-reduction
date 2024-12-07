import tensorflow as tf
from tensorflow.keras.layers import (
    Layer,
    Input,
    Conv2D,
    BatchNormalization,
    Reshape,
    Permute,
    LSTM,
    Dense,
    Lambda,
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


# 自定义 SelectiveBinaryCrossentropy 层
class SelectiveBinaryCrossentropy(Layer):
    def __init__(self, **kwargs):
        super(SelectiveBinaryCrossentropy, self).__init__(**kwargs)

    def call(self, inputs):
        input_data, y_true, y_pred = inputs
        mask = tf.reduce_sum(input_data, axis=-1) > 0  # shape: (batch_size, 64, 128)
        mask = tf.expand_dims(mask, axis=-1)  # Expand to (batch_size, 64, 128, 1)
        y_true_filtered = tf.boolean_mask(y_true, mask)
        y_pred_filtered = tf.boolean_mask(y_pred, mask)
        loss = tf.keras.losses.binary_crossentropy(y_true_filtered, y_pred_filtered)
        return tf.reduce_mean(loss)


# 自定义损失函数
def custom_loss(y_true, y_pred):
    input_data = y_true[..., 1:]  # 从标签中获取附加的输入数据部分
    y_true_only = y_true[..., :1]  # 提取真实标签部分
    selective_bce_layer = SelectiveBinaryCrossentropy()
    return selective_bce_layer([input_data, y_true_only, y_pred])


# 创建自定义层来实现只在非零位置上计算准确率
class SelectiveAccuracy(Layer):
    def __init__(self, **kwargs):
        super(SelectiveAccuracy, self).__init__(**kwargs)

    def call(self, inputs):
        input_data, y_true, y_pred = inputs

        # 生成 mask 以标记非零位置
        mask = tf.reduce_sum(input_data, axis=-1) > 0  # shape: (batch_size, 64, 128)
        mask = tf.expand_dims(mask, axis=-1)  # 扩展到 (batch_size, 64, 128, 1)

        # 使用 mask 筛选出非零位置
        y_true_filtered = tf.boolean_mask(y_true, mask)
        y_pred_filtered = tf.boolean_mask(y_pred, mask)

        # 计算筛选后的准确率
        accuracy = tf.reduce_mean(
            tf.cast(tf.equal(y_true_filtered, tf.round(y_pred_filtered)), tf.float32)
        )
        return accuracy


# 包装成一个指标函数
def custom_accuracy(y_true, y_pred):
    input_data = y_true[..., 1:]  # 从标签中获取附加的输入数据部分
    y_true_only = y_true[..., :1]  # 提取真实标签部分
    selective_accuracy_layer = SelectiveAccuracy()
    return selective_accuracy_layer([input_data, y_true_only, y_pred])


# 简单 Attention 层
def simple_attention(inputs, queries):
    attention = tf.keras.layers.Attention()([queries, inputs])
    return attention


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

        # 将标签和输入数据结合，以便在 custom_loss 中使用
        Y_combined = np.concatenate([Y_batch, X_batch], axis=-1)
        return X_batch, Y_combined

    def on_epoch_end(self):
        np.random.shuffle(self.indices)


# 路径设置
X_train_path = "/home/wanchichang/piano-reduction/LOP_database/X_train.npy"
Y_train_path = "/home/wanchichang/piano-reduction/LOP_database/Y_train.npy"

# 定义 EarlyStopping 和 ModelCheckpoint 回调
output_folder = "my_model_1207-32-lstm03"
early_stopping = EarlyStopping(
    monitor="val_loss", patience=5, verbose=1, restore_best_weights=True
)
# checkpoint = ModelCheckpoint(
#     filepath=f"{output_folder}/best_model.h5",
#     monitor="val_loss",
#     save_best_only=True,
#     verbose=1,
# )
checkpoint = ModelCheckpoint(
    filepath=f"{output_folder}/best_model.keras",  # 修改扩展名为 .keras
    monitor="val_loss",
    save_best_only=True,
    verbose=1,
)
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
# 模型定义
    input_shape = (64, 128, 4)
    inputs = Input(shape=input_shape)

    x = Conv2D(
        16,
        (3, 3),
        activation="relu",
        padding="same",
        kernel_regularizer=regularizers.l1(0.00),
    )(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(
        32,
        (3, 3),
        activation="relu",
        padding="same",
        kernel_regularizer=regularizers.l1(0.00),
    )(x)
    x = BatchNormalization()(x)
    x = Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same",
        kernel_regularizer=regularizers.l1(0.00),
    )(x)
    x = BatchNormalization()(x)

    x = Reshape((64, 128 * 64))(x)  # Flatten keeping the sequence dimension
    x = Permute((2, 1))(x)  # Transpose for attention
    x = simple_attention(x, x)
    x = Permute((2, 1))(x)  # Transpose back

    x = LSTM(
        units=512,
        activation="tanh",
        recurrent_activation="sigmoid",
        dropout=0.3,
        recurrent_dropout=0.3,
        return_sequences=True,
    )(x)

    x = TimeDistributed(Dense(128, activation="sigmoid"))(x)  # Predicting for each pitch
    outputs = Reshape((64, 128, 1))(x)  # Adjust to match label shape

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=Adam(learning_rate=0.001), loss=custom_loss, metrics=[custom_accuracy]
    )
    model.summary()
    save_path = f"/home/wanchichang/piano-reduction/code/{output_folder}"

    # 确保目录存在
    os.makedirs(save_path, exist_ok=True)

    # 训练模型
    batch_size = 32
    validation_split = 0.2
    epochs = 30

    train_generator = DataGenerator(
        X_train_path, Y_train_path, batch_size, validation_split, is_validation=False
    )
    validation_generator = DataGenerator(
        X_train_path, Y_train_path, batch_size, validation_split, is_validation=True
    )

    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=[early_stopping, checkpoint],
    )
    model.save(output_folder, save_format="tf")
# plot

plt.figure(figsize=(8, 5))
# 创建从 1 开始的 epochs 数组
epochs = np.arange(1, len(history.history["loss"]) + 1)

# 画图时，需要让 x 轴从 1 开始，数据也从 1 对应
plt.plot(epochs, history.history["loss"], label="Training Loss")
plt.plot(epochs, history.history["val_loss"], label="Validation Loss")

plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

# 设置 x 轴刻度
plt.xticks(epochs)  # 将x坐标设为从1开始

plt.legend()
plt.savefig(
    f"/home/wanchichang/piano-reduction/code/{output_folder}/training_validation_loss.png"
)
plt.close()

# 繪製和保存 Accuracy 圖像
plt.figure(figsize=(8, 5))

# 创建从 1 开始的 epochs 数组
epochs = np.arange(1, len(history.history["custom_accuracy"]) + 1)

# 画图时，需要让 x 轴从 1 开始，数据也从 1 对应
plt.plot(epochs, history.history["custom_accuracy"], label="Training Accuracy")
plt.plot(epochs, history.history["val_custom_accuracy"], label="Validation Accuracy")

plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

# 设置 x 轴刻度
plt.xticks(epochs)  # 将x坐标设为从1开始

plt.legend()
plt.savefig(
    f"/home/wanchichang/piano-reduction/code/{output_folder}/training_validation_accuracy.png"
)
plt.close()
