import os

# 目標資料夾路徑
target_folder = "../LOP_database/aligned"

# 存儲曲名及其矩陣數量的列表
songs_info = []

# 遍歷目標資料夾下的每個資料夾
for root, dirs, files in os.walk(target_folder):
    for folder in dirs:
        folder_path = os.path.join(root, folder)
        info_file_path = os.path.join(folder_path, "info.txt")

        if os.path.exists(info_file_path):
            with open(info_file_path, "r") as f:
                first_line = f.readline().strip()
                if first_line:
                    parts = first_line.split("- Number of Phrases: ")
                    if len(parts) == 2:
                        num_matrices = int(parts[1])
                        songs_info.append((folder, num_matrices))

# 按矩陣數量從大到小排序
sorted_songs_info = sorted(songs_info, key=lambda x: x[1], reverse=True)

# 輸出結果到一個新的文本文件
output_file_path = os.path.join(target_folder, "sorted_songs.txt")
with open(output_file_path, "w") as f:
    for song in sorted_songs_info:
        f.write(f"{song[0]}: {song[1]}\n")

print(f"結果已輸出到 {output_file_path}")
