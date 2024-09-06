import tensorflow as tf


def within_range(tensor, k):
    num_pitches = tensor.shape[0]

    # 计算最大和最小音高
    max_pitch = tf.reduce_max(
        tf.where(tensor > 0, tf.range(num_pitches, dtype=tf.float32), -1)
    )
    min_pitch = tf.reduce_min(
        tf.where(tensor > 0, tf.range(num_pitches, dtype=tf.float32), num_pitches)
    )

    # 检查从 max_pitch 向下 k 个位置的范围
    down_start = tf.maximum(max_pitch - k, 0)
    down_end = max_pitch + 1
    down_range_indices = tf.range(down_start, down_end, dtype=tf.int32)
    down_range_mask = tf.scatter_nd(
        tf.expand_dims(down_range_indices, axis=1),
        tf.ones_like(down_range_indices, dtype=tf.float32),
        [num_pitches],
    )

    # 检查从 min_pitch 向上 k 个位置的范围
    up_start = min_pitch
    up_end = tf.minimum(min_pitch + k + 1, num_pitches)
    up_range_indices = tf.range(up_start, up_end, dtype=tf.int32)
    up_range_mask = tf.scatter_nd(
        tf.expand_dims(up_range_indices, axis=1),
        tf.ones_like(up_range_indices, dtype=tf.float32),
        [num_pitches],
    )

    # 生成范围掩码
    down_range_mask = tf.cast(down_range_mask, dtype=tf.bool)
    up_range_mask = tf.cast(up_range_mask, dtype=tf.bool)

    # 合并两个范围掩码
    combined_range_mask = tf.logical_or(down_range_mask, up_range_mask)

    # 检查是否有音符超出范围
    tensor_mask = tf.cast(tensor, dtype=tf.bool)
    out_of_range_mask = tf.logical_and(tensor_mask, tf.logical_not(combined_range_mask))

    # 判断是否有超出范围的音符
    out_of_range = tf.reduce_any(out_of_range_mask)

    return out_of_range


# # 示例使用
# tensor = tf.constant([0, 1, 0, 1, 0, 0, 1], dtype=tf.float32)
# max_pitch = 3
# min_pitch = 2
# k = 2

# print("Any pitch out of range:", within_range(tensor, max_pitch, min_pitch, k).numpy())


# # 定义 calculate_hand_span 函数
# def calculate_hand_span(y_pred, k1=8, k2=10):
#     batch_size, timesteps, num_pitches, _ = y_pred.shape
#     max_hand_spans = []

#     for i in range(batch_size):
#         max_hand_span_for_batch = tf.constant(0.0, dtype=tf.float32)

#         for t in range(timesteps):
#             timestep_data = y_pred[i, t, :, 0]  # (num_pitches,)
#             indices = tf.range(num_pitches, dtype=tf.float32)

#             # 计算当前时间步的最大音高和最小音高
#             max_pitch = tf.reduce_max(tf.where(timestep_data > 0, indices, -1))
#             min_pitch = tf.reduce_min(tf.where(timestep_data > 0, indices, num_pitches))

#             # 检查 handspan <= 6
#             out_of_range_k1 = within_range(timestep_data, k1)
#             handspan_for_timestep = tf.cond(
#                 tf.logical_not(out_of_range_k1),
#                 lambda: tf.constant(6.0, dtype=tf.float32),
#                 lambda: tf.cond(
#                     tf.logical_not(within_range(timestep_data, k2)),
#                     lambda: tf.constant(10.0, dtype=tf.float32),
#                     lambda: tf.constant(11.0, dtype=tf.float32),  # handspan > 10
#                 ),
#             )

#             # 更新当前 batch 的最大 handspan
#             max_hand_span_for_batch = tf.maximum(
#                 max_hand_span_for_batch, handspan_for_timestep
#             )

#         # 根据最大手跨度为 batch 计算分数
#         handspan_score = tf.cond(
#             max_hand_span_for_batch <= 6,
#             lambda: tf.constant(0.0, dtype=tf.float32),
#             lambda: tf.cond(
#                 max_hand_span_for_batch <= 10,
#                 lambda: tf.constant(0.5, dtype=tf.float32),
#                 lambda: tf.constant(1.0, dtype=tf.float32),
#             ),
#         )

#         max_hand_spans.append(handspan_score)

#     # 计算所有 batch 的平均 handspan score
#     avg_hand_span_score = tf.reduce_mean(max_hand_spans)
#     return avg_hand_span_score


def calculate_hand_span(y_pred, k1=8, k2=10):
    batch_size, timesteps, num_pitches, _ = y_pred.shape
    max_hand_spans = []

    for i in range(batch_size):
        max_hand_span_for_batch = tf.constant(0.0, dtype=tf.float32)

        for t in range(timesteps):
            timestep_data = y_pred[i, t, :, 0]  # (num_pitches,)
            indices = tf.range(num_pitches, dtype=tf.float32)

            max_pitch = tf.reduce_max(tf.where(timestep_data > 0, indices, -1))
            min_pitch = tf.reduce_min(tf.where(timestep_data > 0, indices, num_pitches))

            # 检查手跨度
            out_of_range_k1 = within_range(timestep_data, k1)
            out_of_range_k2 = within_range(timestep_data, k2)

            # 计算当前时间步的手跨度
            handspan_for_timestep = tf.where(
                tf.logical_not(out_of_range_k1),
                6.0,
                tf.where(tf.logical_not(out_of_range_k2), 10.0, 11.0),
            )

            max_hand_span_for_batch = tf.maximum(
                max_hand_span_for_batch, handspan_for_timestep
            )

        # 根据最大手跨度为 batch 计算分数
        handspan_score = tf.where(
            max_hand_span_for_batch <= 6,
            0.0,
            tf.where(max_hand_span_for_batch <= 10, 0.5, 1.0),
        )

        max_hand_spans.append(handspan_score)

    avg_hand_span_score = tf.reduce_mean(max_hand_spans)
    return avg_hand_span_score


# def calculate_num_of_notes_score(y_pred):
#     batch_size, timesteps, num_pitches, _ = y_pred.shape

#     # 初始化一个张量来存储每个batch的最大音符数量
#     max_num_of_notes_per_batch = tf.zeros((batch_size,), dtype=tf.float32)

#     for t in range(timesteps):
#         # 提取当前时间步的数据
#         timestep_data = y_pred[:, t, :, :]  # 形状: (batch_size, num_pitches, 1)

#         # 计算当前时间步的音符数量
#         num_of_notes = tf.reduce_sum(timestep_data, axis=[1, 2])  # 形状: (batch_size,)

#         # 更新每个batch的最大音符数量
#         max_num_of_notes_per_batch = tf.maximum(
#             max_num_of_notes_per_batch, num_of_notes
#         )

#     # 计算所有batch的最大num_of_notes的平均值
#     avg_max_num_of_notes = tf.reduce_mean(max_num_of_notes_per_batch)

#     # 根据平均值判断num_of_notes_score
#     num_of_notes_score = tf.cond(
#         avg_max_num_of_notes > 10,
#         lambda: tf.constant(0.0, dtype=tf.float32),
#         lambda: tf.cond(
#             avg_max_num_of_notes > 6,
#             lambda: tf.constant(0.5, dtype=tf.float32),
#             lambda: tf.constant(1.0, dtype=tf.float32),
#         ),
#     )

#     return num_of_notes_score


def calculate_num_of_notes_score(y_pred):
    batch_size, timesteps, num_pitches, _ = y_pred.shape

    max_num_of_notes_per_batch = tf.zeros((batch_size,), dtype=tf.float32)

    for t in range(timesteps):
        timestep_data = y_pred[:, t, :, :]  # 形状: (batch_size, num_pitches, 1)
        num_of_notes = tf.reduce_sum(timestep_data, axis=[1, 2])  # 形状: (batch_size,)
        max_num_of_notes_per_batch = tf.maximum(
            max_num_of_notes_per_batch, num_of_notes
        )

    avg_max_num_of_notes = tf.reduce_mean(max_num_of_notes_per_batch)

    num_of_notes_score = tf.where(
        avg_max_num_of_notes > 10, 0.0, tf.where(avg_max_num_of_notes > 6, 0.5, 1.0)
    )

    return num_of_notes_score


def calculate_playability_score(y_pred):
    # 计算手跨度得分
    hand_span_score = calculate_hand_span(y_pred)

    # 计算音符数量得分
    num_of_notes_score = calculate_num_of_notes_score(y_pred)

    # 计算最终的可演奏分数
    playability_score = hand_span_score * 0.5 + num_of_notes_score * 0.5

    return playability_score


def custom_loss_with_playability(y_true, y_pred, lambda_param=0.1):
    # 二元交叉熵音乐损失
    musical_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    musical_loss = tf.nn.sigmoid(musical_loss)
    # 计算可演奏性得分 P(x)
    playability_score = tf.numpy_function(
        lambda y_pred: calculate_playability_score(y_pred), [y_pred], tf.float32
    )

    # 计算最终损失 L
    # loss = (1 - lambda_param) * musical_loss + lambda_param * (1 - playability_score)
    loss = musical_loss + lambda_param * (1 - playability_score)
    return loss
