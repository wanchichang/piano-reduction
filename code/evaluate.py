import tensorflow as tf
import numpy as np
import os
from playability import (
    custom_loss_with_playability,
)  # 确保自定义损失函数已经正确定义并导入
from tensorflow.keras import backend as K

# from model import binary_threshold


def binary_threshold(x):
    return tf.where(x >= 0.5, 1.0, 0.0)


model_folder = "my_model_0923"

# 加载保存的模型，并指定自定义损失函数
model = tf.keras.models.load_model(
    model_folder,
    # custom_objects={"loss": custom_loss_with_playability},
    compile=False,
)
model.compile(loss=custom_loss_with_playability, optimizer="adam", metrics="accuracy")

# 加载测试数据
X_test = np.load("/home/wanchichang/piano-reduction/LOP_database/X_test.npy")
Y_test = np.load("/home/wanchichang/piano-reduction/LOP_database/Y_test.npy")
X_test = np.transpose(X_test, (0, 2, 3, 1))
Y_test = np.transpose(Y_test, (0, 2, 3, 1))

# 评估模型
loss, accuracy = model.evaluate(X_test, Y_test, batch_size=16, verbose=1)

# 打印结果
print(f"Test loss: {loss}")
print(f"Test accuracy: {accuracy}")

# 指定保存路径
save_path = f"/home/wanchichang/piano-reduction/code/{model_folder}"

# 确保目录存在
os.makedirs(save_path, exist_ok=True)

# 将结果保存到文本文件
results_file_path = os.path.join(save_path, "evaluation_results.txt")
with open(results_file_path, "w") as f:
    f.write(f"Test loss: {loss}\n")
    f.write(f"Test accuracy: {accuracy}\n")

print(f"评估结果已保存到 {results_file_path}")
