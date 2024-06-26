import os
import numpy as np
from LOP_database.midi.read_midi import Read_midi

# 目標資料夾路徑
target_folder = "../LOP_database/aligned"

# 遍歷目標資料夾下的每個資料夾
for root, dirs, files in os.walk(target_folder):
    for folder in dirs:
        folder_path = os.path.join(root, folder)
        print("Processing folder:", folder_path)

        # 在每個資料夾中尋找 MIDI 檔案
        for file in os.listdir(folder_path):
            if file.endswith(".mid"):
                midi_file_path = os.path.join(folder_path, file)

                # 實例化 Read_midi 類別，傳遞 MIDI 檔案路徑和所需的量化參數
                midi_reader = Read_midi(midi_file_path, quantization=4)

                # 讀取 MIDI 檔案並獲取 pianoroll 資料
                pianoroll_dict = midi_reader.read_file()

                # 輸出的檔案名稱為原始 MIDI 檔案的檔名，副檔名改為 .txt
                output_file_name = os.path.splitext(file)[0] + ".txt"
                output_file_path = os.path.join(folder_path, output_file_name)
                num_matrices_dict = {}

                # 寫入 pianoroll 切割成多個矩陣後的內容到文件中
                with open(output_file_path, "w") as f:
                    for track_name, pianoroll in pianoroll_dict.items():
                        f.write(f"Track Name: {track_name}\n")
                        f.write("Pianoroll:\n")

                        # 定義切割的行數
                        # 4/4 拍, 1/16音符為單位,  1 measure = (1/4)/(1/16)*4 = 16
                        # 1 phrase = 4 measures = 64 timesteps

                        rows_per_phrase = 64

                        num_rows = len(pianoroll)
                        num_matrices = (
                            num_rows + rows_per_phrase - 1
                        ) // rows_per_phrase  # 向上取整計算矩陣數量
                        num_matrices_dict[track_name] = num_matrices

                        # 將 pianoroll 切割成多個矩陣
                        # num_matrices = len(pianoroll) // rows_per_phrase
                        for i in range(num_matrices):
                            # 計算要切割的起始和結束行索引
                            start_index = i * rows_per_phrase
                            end_index = (i + 1) * rows_per_phrase
                            # 取出部分 pianoroll
                            sub_pianoroll = np.zeros(
                                (rows_per_phrase, pianoroll.shape[1]), dtype=int
                            )
                            sub_pianoroll[
                                : min(rows_per_phrase, num_rows - start_index), :
                            ] = pianoroll[start_index:end_index, :]
                            sub_pianoroll[sub_pianoroll > 0] = 1
                            # 將 sub_pianoroll 寫入文件
                            f.write(f"Phrase {i+1}:\n")
                            for row in sub_pianoroll:
                                f.write(",".join(map(str, row)) + "\n")
                            f.write("\n")

                # 創建並寫入 info.txt 文件
                info_file_path = os.path.join(folder_path, "info.txt")
                with open(info_file_path, "w") as info_file:
                    for track_name, num_phrase in num_matrices_dict.items():
                        info_file.write(
                            f"Track Name: {track_name} - Number of Phrases: {num_phrase}\n"
                        )
                        # # print out non-zero values in the matrix
                        # print(
                        #     f"Non-zero values in Matrix {i+1} for Track {track_name}:"
                        # )
                        # for r_idx, row in enumerate(sub_pianoroll):
                        #     for c_idx, value in enumerate(row):
                        #         if value != 0:
                        #             print(
                        #                 f"Row: {r_idx}, Column: {c_idx}, Value: {value}"
                        #             )
