# from util import getMeasure, getTrackNumber
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import Sequence, plot_model
import os
from tensorflow.keras import regularizers
from playability import custom_loss_with_playability
from tensorflow.keras import backend as K


def binary_threshold(x):
    return tf.where(x >= 0.5, 1.0, 0.0)


def debug_loss(y_true, y_pred):
    # 打印 y_true 和 y_pred 的形状
    tf.print("y_true shape:", tf.shape(y_true))
    tf.print("y_pred shape:", tf.shape(y_pred))

    # 计算一个简单的损失，这里使用的是均方误差（MSE）作为示例
    return tf.keras.losses.mean_squared_error(y_true, y_pred)


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

output_folder = "plt_test"
# 定义 EarlyStopping 回调
early_stopping = EarlyStopping(
    monitor="val_loss",  # 监控验证集损失
    patience=5,  # 如果在 5 个 epoch 内没有改善，则停止训练
    verbose=1,  # 打印训练过程中回调的日志
    restore_best_weights=True,  # 恢复具有最佳验证集损失的模型权重
)

# 其他必要的回调（如保存模型检查点）
checkpoint = ModelCheckpoint(
    filepath=f"{output_folder}/best_model.h5",  # 保存最佳模型的路径
    monitor="val_loss",  # 监控验证集损失
    save_best_only=True,  # 仅保存最好的模型
    verbose=1,  # 打印模型保存的日志
)

params = {
    "batch_size": 16,
    "epochs": 20,
    "learning_rate": 0.001,
    "input_shape": (64, 128, 4),
}

batch_size = params["batch_size"]
validation_split = 0.2
epochs = params["epochs"]

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

x = tf.keras.layers.Conv2D(
    16,
    (3, 3),
    activation="relu",
    padding="same",
    kernel_regularizer=regularizers.l1(0.00),
)(inputs)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Conv2D(
    32,
    (3, 3),
    activation="relu",
    padding="same",
    kernel_regularizer=regularizers.l1(0.00),
)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Conv2D(
    64,
    (3, 3),
    activation="relu",
    padding="same",
    kernel_regularizer=regularizers.l1(0.00),
)(x)
x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.Reshape((64, 128 * 64))(x)  # Flatten keeping the sequence dimension

x = tf.keras.layers.Permute((2, 1))(x)  # Transpose for attention
x = simple_attention(x, x)
x = tf.keras.layers.Permute((2, 1))(x)  # Transpose back

# x = tf.keras.layers.LSTM(512, return_sequences=True)(x)

# Adding parameters to the LSTM layer
x = tf.keras.layers.LSTM(
    units=512,  # 输出维度
    activation="tanh",  # 激活函数
    recurrent_activation="sigmoid",  # 递归激活函数
    dropout=0.4,  # 输入丢弃比例
    recurrent_dropout=0.0,  # 递归状态丢弃比例
    return_sequences=True,  # 是否返回输出序列中的每个输出
    return_state=False,  # 是否返回最后一个状态
    go_backwards=False,  # 是否反向处理输入序列
    stateful=False,  # 是否使用有状态 LSTM
)(x)


# x = tf.keras.layers.Reshape((64, 512))(x)

# 改变输出层，使其输出形状为 (64, 128, 1)
x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(128, activation="sigmoid"))(
    x
)  # Predicting for each pitch
x = tf.keras.layers.Lambda(lambda z: K.in_train_phase(z, binary_threshold(z)))(x)
# x = tf.keras.layers.Lambda(binary_threshold)(x)  # Apply binary thresholding
outputs = tf.keras.layers.Reshape((64, 128, 1))(x)  # Adjust to match the label shape

model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
# model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.compile(
    optimizer="adam",
    loss=lambda y_true, y_pred: custom_loss_with_playability(
        y_true, y_pred, lambda_param=0.6
    ),
    metrics=["accuracy"],
)

model.summary()


# 指定保存路径
save_path = f"/home/wanchichang/piano-reduction/code/{output_folder}"

# 确保目录存在
os.makedirs(save_path, exist_ok=True)

plot_model(
    model,
    to_file=f"{save_path}/model_plot.png",
    show_shapes=True,
    show_layer_names=True,
)
optimizer = model.optimizer

# 获取当前学习率
current_learning_rate = optimizer.learning_rate.numpy()
print(f"Current learning rate: {current_learning_rate}")
history = model.fit(
    train_generator,
    epochs=epochs,  # 最大训练 epoch 数
    validation_data=validation_generator,
    callbacks=[early_stopping],  # 使用 EarlyStopping 回调
)
# history = model.fit(train_generator, epochs=epochs, validation_data=validation_generator)
model.save(output_folder, save_format="tf")


# 保存超参数到文本文件
params_file_path = os.path.join(save_path, "model_params.txt")
with open(params_file_path, "w") as f:
    for key, value in params.items():
        f.write(f"{key}: {value}\n")

print(f"超参数已保存到 {params_file_path}")


# plot
import matplotlib.pyplot as plt

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
epochs = np.arange(1, len(history.history["accuracy"]) + 1)

# 画图时，需要让 x 轴从 1 开始，数据也从 1 对应
plt.plot(epochs, history.history["accuracy"], label="Training Accuracy")
plt.plot(epochs, history.history["val_accuracy"], label="Validation Accuracy")

plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Loss")

# 设置 x 轴刻度
plt.xticks(epochs)  # 将x坐标设为从1开始

plt.legend()
plt.savefig(
    f"/home/wanchichang/piano-reduction/code/{output_folder}/training_validation_accuracy.png"
)
plt.close()


def create_inference_function(model):
    input_tensor = model.input
    output_tensor = model.output
    func = tf.keras.backend.function([input_tensor], [output_tensor])
    return func


# 获取模型的预测函数
inference_func = create_inference_function(model)

# 生成一个示例输入
example_input = np.random.randint(0, 2, size=(1, 64, 128, 4), dtype=np.int32)

# 使用模型进行预测
example_output = inference_func([example_input])[0]
np.set_printoptions(threshold=np.inf)  # 设置为打印所有内容
# 打印示例输入和输出
# print("Example Input:")
# print(example_input)

# print("Example Output:")
# print(example_output)
import os

with open("output.txt", "w") as f:
    f.write("Example Input:\n")
    np.savetxt(
        f, example_input.flatten(), fmt="%d", delimiter=",", header="Flat Input Array"
    )
    f.write("\nExample Output:\n")
    np.savetxt(
        f,
        example_output.flatten(),
        fmt="%.5f",
        delimiter=",",
        header="Flat Output Array",
    )

print("Results saved to output.txt")
