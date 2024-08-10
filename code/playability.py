import tensorflow as tf


def within_range(tensor, lower_bound, upper_bound):
    return tf.logical_and(
        tf.greater_equal(tensor, lower_bound), tf.less_equal(tensor, upper_bound)
    )


# def calculate_hand_span(timestep_data, k=10):
#     """Calculate the hand span for each timestep."""
#     print("timestep_data shape:", timestep_data.shape)  # Debug print
#     batch_size, num_pitches, _ = timestep_data.shape
#     # batch_size, num_pitches = timestep_data.shape
#     hand_spans = []

#     for i in range(batch_size):
#         notes = tf.where(timestep_data[i, :, 0] > 0)[:, 0]
#         if tf.size(notes) == 0:
#             hand_spans.append(0)
#             continue

#         min_note = tf.reduce_min(notes)
#         max_note = tf.reduce_max(notes)

#         # Check if all notes are within the hand span range
#         right_range = tf.range(max_note, min_note - k, -1)
#         left_range = tf.range(min_note, max_note + k)

#         right_within = tf.reduce_any(
#             tf.reduce_any(tf.isin(timestep_data[i, :, 0], right_range), axis=-1)
#         )
#         left_within = tf.reduce_any(
#             tf.reduce_any(tf.isin(timestep_data[i, :, 0], left_range), axis=-1)
#         )

#         if right_within and left_within:
#             hand_spans.append(max_note - min_note)
#         else:
#             hand_spans.append(tf.float32.max)  # Exceeds hand span limit

#     return tf.reduce_max(hand_spans)


def calculate_hand_span(timestep_data, k):
    batch_size, num_pitches, _ = timestep_data.shape

    hand_spans = []
    for i in range(batch_size):
        # 提取當前時間點的數據
        timestep_data_i = timestep_data[i, :, 0]

        # 計算最大音高和最小音高
        max_pitch = tf.reduce_max(
            tf.where(timestep_data_i > 0, tf.range(num_pitches, dtype=tf.float32), -1)
        )
        min_pitch = tf.reduce_min(
            tf.where(
                timestep_data_i > 0,
                tf.range(num_pitches, dtype=tf.float32),
                num_pitches,
            )
        )

        # 計算可彈奏的範圍
        right_range = tf.range(tf.minimum(max_pitch + k, num_pitches))
        left_range = tf.range(tf.maximum(min_pitch - k, 0), num_pitches)

        # 檢查音高是否在範圍內
        right_within_range = within_range(
            tf.range(num_pitches, dtype=tf.float32), max_pitch - k, max_pitch
        )
        left_within_range = within_range(
            tf.range(num_pitches, dtype=tf.float32), min_pitch, min_pitch + k
        )

        # 判斷是否有音符超出範圍
        right_violations = tf.reduce_any(tf.logical_not(right_within_range))
        left_violations = tf.reduce_any(tf.logical_not(left_within_range))

        # 根據是否有超出範圍的音符計算音程
        max_interval = tf.cond(
            right_violations or left_violations,
            lambda: tf.maximum(max_pitch - min_pitch, 0),
            lambda: 0,
        )

        hand_spans.append(max_interval)

    hand_span_mean = tf.reduce_mean(hand_spans)
    return hand_span_mean


# def calculate_playability_score(y_pred, k=10):
#     batch_size, timesteps, num_pitches, _ = y_pred.shape
#     max_num_of_notes = 0
#     max_hand_span = 0

#     for t in range(timesteps):
#         timestep_data = y_pred[:, t, :, :]  # Shape: (batch_size, num_pitches, 1)

#         # Calculate num_of_notes for this timestep
#         num_of_notes = tf.reduce_sum(timestep_data, axis=[1, 2])  # Shape: (batch_size,)
#         max_num_of_notes = tf.maximum(
#             max_num_of_notes, tf.reduce_max(num_of_notes)
#         )  # Max across all timesteps

#         # Calculate hand_span for this timestep
#         max_interval = calculate_hand_span(timestep_data, k)

#         # Update max_hand_span
#         max_hand_span = tf.maximum(max_hand_span, max_interval)

#     # Compute HandSpan and NumOfNotes score
#     hand_span_score = tf.cond(
#         max_hand_span <= 6,
#         lambda: 1.0,
#         lambda: tf.cond(max_hand_span <= 10, lambda: 0.5, lambda: 0.0),
#     )

#     num_of_notes_score = tf.cond(
#         max_num_of_notes > 10,
#         lambda: 0.0,
#         lambda: tf.cond(max_num_of_notes > 6, lambda: 0.5, lambda: 1.0),
#     )

#     playability_score = hand_span_score * 0.5 + num_of_notes_score * 0.5


#     return playability_score
def calculate_playability_score(y_pred, k=10):
    batch_size, timesteps, num_pitches, _ = y_pred.shape
    max_num_of_notes = tf.constant(0.0, dtype=tf.float32)
    max_hand_span = tf.constant(0.0, dtype=tf.float32)

    for t in range(timesteps):
        timestep_data = y_pred[:, t, :, :]  # 形状: (batch_size, num_pitches, 1)

        # 计算当前时间步的音符数量
        num_of_notes = tf.reduce_sum(timestep_data, axis=[1, 2])  # 形状: (batch_size,)
        max_num_of_notes = tf.maximum(
            max_num_of_notes, tf.reduce_max(num_of_notes)
        )  # 在所有时间步中取最大值

        # 计算当前时间步的手跨度
        max_interval = calculate_hand_span(timestep_data, k)

        # 更新最大手跨度
        max_hand_span = tf.maximum(max_hand_span, max_interval)

    # 计算手跨度和音符数量得分
    hand_span_score = tf.cond(
        max_hand_span <= 6,
        lambda: tf.constant(1.0, dtype=tf.float32),
        lambda: tf.cond(
            max_hand_span <= 10,
            lambda: tf.constant(0.5, dtype=tf.float32),
            lambda: tf.constant(0.0, dtype=tf.float32),
        ),
    )

    num_of_notes_score = tf.cond(
        max_num_of_notes > 10,
        lambda: tf.constant(0.0, dtype=tf.float32),
        lambda: tf.cond(
            max_num_of_notes > 6,
            lambda: tf.constant(0.5, dtype=tf.float32),
            lambda: tf.constant(1.0, dtype=tf.float32),
        ),
    )

    playability_score = hand_span_score * 0.5 + num_of_notes_score * 0.5

    return playability_score


def custom_loss_with_playability(y_true, y_pred, lambda_param=0.1):
    # 二元交叉熵音乐损失
    musical_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)

    # 计算可演奏性得分 P(x)
    playability_score = tf.numpy_function(
        lambda y_pred: calculate_playability_score(y_pred), [y_pred], tf.float32
    )

    # 计算最终损失 L
    loss = musical_loss + lambda_param * (1 - playability_score)
    return loss
