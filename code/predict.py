import tensorflow as tf
import numpy as np
import os
from playability import (
    custom_loss_with_playability,
)  # 确保自定义损失函数已经正确定义并导入
from tensorflow.keras import backend as K
from LOP_database.midi.write_midi import write_midi

# from model import binary_threshold


def binary_threshold(x):
    return tf.where(x >= 0.5, 1.0, 0.0)


model_folder = "my_model_0925"

# 加载保存的模型，并指定自定义损失函数
model = tf.keras.models.load_model(
    model_folder,
    compile=False,
)
model.compile(loss=custom_loss_with_playability, optimizer="adam", metrics="accuracy")

# 加载测试数据
X_test = np.load("/home/wanchichang/piano-reduction/LOP_database/demo/music/X_test.npy")
# X_test = np.load("/home/wanchichang/piano-reduction/LOP_database/test/X_test2.npy")
# Y_test = np.load("/home/wanchichang/piano-reduction/LOP_database/Y_test2.npy")
X_test = np.transpose(X_test, (0, 2, 3, 1))
# Y_test = np.transpose(Y_test, (0, 2, 3, 1))

# 获取一笔测试数据
X_sample = X_test[0:1]  # 取出第一个样本
# Y_sample = Y_test[0:1]  # 对应的真实标签

# 使用模型进行预测
prediction = model.predict(X_sample)
print(prediction.shape)
# 二值化处理预测结果
# prediction_binary = binary_threshold(prediction)

# 指定保存路径
save_path = f"/home/wanchichang/piano-reduction/LOP_database/demo"

# 确保目录存在
os.makedirs(save_path, exist_ok=True)

# 保存原始标签、预测结果（经过二值化处理前后）到 .npy 文件
# np.save(os.path.join(save_path, "Y_sample.npy"), Y_sample)
np.save(os.path.join(save_path, "prediction.npy"), prediction)
# np.save(os.path.join(save_path, "prediction_binary.npy"), prediction_binary)

print(f"预测结果已保存到 {save_path}")

# prediction_np = K.eval(prediction).flatten()
# Y_sample_np = K.eval(Y_sample).flatten()

# # 计算正确的预测数量
# correct_predictions = np.sum(prediction_np == Y_sample_np)
# # 计算总样本数量
# total_samples = Y_sample_np.size
# # 计算准确率
# accuracy = correct_predictions / total_samples

# # 输出准确率
# print(f"Accuracy between prediction and ground truth: {accuracy:.4f}")
