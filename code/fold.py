import os
import random

# 讀取 sorted_songs.txt
sorted_songs_file = "../LOP_database/aligned/sorted_songs.txt"

songs_info = []
with open(sorted_songs_file, "r") as f:
    for line in f:
        parts = line.strip().split(": ")
        if len(parts) == 2:
            song_name = parts[0]
            num_matrices = int(parts[1])
            songs_info.append((song_name, num_matrices))

# 將曲子按矩陣數量分組
songs_info.sort(key=lambda x: x[1])

# 初始化5個fold
folds = [[] for _ in range(5)]

# 平均分配曲子到5個fold
for i, song in enumerate(songs_info):
    folds[i % 5].append(song)

# 打亂每個fold中的曲子順序
for fold in folds:
    random.shuffle(fold)

# 保存每個fold中的曲子到新的文件
output_folder = "../LOP_database/aligned/folds"
os.makedirs(output_folder, exist_ok=True)

for i, fold in enumerate(folds):
    fold_file = os.path.join(output_folder, f"fold_{i + 1}.txt")
    with open(fold_file, "w") as f:
        for song in fold:
            f.write(f"{song[0]}: {song[1]}\n")

print(f"每個fold的曲子已經保存到 {output_folder} 中")
