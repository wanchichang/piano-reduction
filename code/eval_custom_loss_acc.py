import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
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

    def get_config(self):
        config = super(SelectiveBinaryCrossentropy, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

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
            tf.cast(
                tf.logical_or(
                    tf.logical_and(tf.less_equal(y_pred_filtered, 0.5), tf.equal(y_true_filtered, 0.0)),
                    tf.logical_and(tf.greater(y_pred_filtered, 0.5), tf.equal(y_true_filtered, 1.0))
                ),
                tf.float32
            )
        )
        return accuracy

    def get_config(self):
        config = super(SelectiveAccuracy, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# 包装成一个指标函数
def custom_accuracy(y_true, y_pred):
    input_data = y_true[..., 1:]  # 从标签中获取附加的输入数据部分
    y_true_only = y_true[..., :1]  # 提取真实标签部分
    selective_accuracy_layer = SelectiveAccuracy()
    return selective_accuracy_layer([input_data, y_true_only, y_pred])

# 測試數據生成器類
class TestDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, X_path, Y_path, batch_size):
        self.X_path = X_path
        self.Y_path = Y_path
        self.batch_size = batch_size
        self.X = np.load(X_path, mmap_mode="r")
        self.Y = np.load(Y_path, mmap_mode="r")
        self.indices = np.arange(self.X.shape[0])

    def __len__(self):
        return int(np.floor(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        X_batch = self.X[batch_indices]
        Y_batch = self.Y[batch_indices]

        # 調整輸入數據的形狀
        X_batch = X_batch.transpose(
            0, 2, 3, 1
        )  # 將形狀從 (batch_size, 4, 64, 128) 調整為 (batch_size, 64, 128, 4)
        Y_batch = Y_batch.transpose(
            0, 2, 3, 1
        )  # 將形狀從 (batch_size, 1, 64, 128) 調整為 (batch_size, 64, 128, 1)

        # 將標籤和輸入數據結合，以便在 custom_loss 中使用
        Y_combined = np.concatenate([Y_batch, X_batch], axis=-1)
        return X_batch, Y_combined

# 測試集路徑
X_test_path = "/home/wanchichang/piano-reduction/LOP_database/X_test.npy"
Y_test_path = "/home/wanchichang/piano-reduction/LOP_database/Y_test.npy"

# 批量大小
batch_size = 32

# 初始化測試數據生成器
test_generator = TestDataGenerator(X_test_path, Y_test_path, batch_size)

# 加載模型
output_folder = "my_model_1217-32-256-learning_rate"
model_path = os.path.join(output_folder, "best_model.keras")
# 設置分佈策略
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    # 加載模型
    model = tf.keras.models.load_model(
        output_folder,
        custom_objects={
            "custom_loss": custom_loss,
            "custom_accuracy": custom_accuracy,
            "SelectiveBinaryCrossentropy": SelectiveBinaryCrossentropy,
            "SelectiveAccuracy": SelectiveAccuracy
        },
        compile=False  # 不編譯模型，手動設定損失函數與指標
    )

    # 手動編譯模型
    model.compile(
        loss=custom_loss,  # 使用您的自定義損失函數
        optimizer="adam",
        metrics=[
            custom_accuracy,  # 您的自定義準確率指標
        ]
    )

# 評估模型
results = model.evaluate(test_generator, verbose=1)

# 打印測試結果
print("Test Loss:", results[0])
print("Test Accuracy:", results[1])
# 使用測試集的第一筆資料進行預測
#first_batch = test_generator[0]  # 取出第一批資料
#X_first = first_batch[0]  # 測試集的輸入
#Y_true_first = first_batch[1][..., :1]  # 測試集的真實標籤部分

# 預測
#Y_pred_first = model.predict(X_first)

# 將預測結果轉化為二進制（0或1）
#Y_pred_first_binary = (Y_pred_first > 0.5).astype(np.float32)

# 保存結果到 .npy 文件
#np.save("X_first.npy", X_first)
#np.save("Y_true_first.npy", Y_true_first)
#np.save("Y_pred_first.npy", Y_pred_first)
#np.save("Y_pred_first_binary.npy", Y_pred_first_binary)

# 打印結果
#print("True Labels for First Sample:", Y_true_first[0])
#print("Predicted Labels (Raw) for First Sample:", Y_pred_first[0])
#print("Predicted Labels (Binary) for First Sample:", Y_pred_first_binary[0])

