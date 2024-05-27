import os

# 定义四类乐器的名称
brass_instruments = ["Horn", "Trumpet", "Trombone", "Tuba"]
woodwinds_with_reeds = ["Oboe", "Clarinet", "Bassoon"]
woodwinds_without_reeds = ["Flute", "Piccolo"]
string_instruments = ["Violin", "Viola", "Violoncello", "Contrabass"]

# 用于存储分类结果的字典
instrument_categories = {
    "brass": set(),
    "woodwinds_with_reeds": set(),
    "woodwinds_without_reeds": set(),
    "strings": set(),
}


# 遍历文件夹并读取 info.txt
def traverse_and_classify_instruments(root_folder):
    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)
        if os.path.isdir(folder_path):
            info_file_path = os.path.join(folder_path, "info.txt")
            if os.path.exists(info_file_path):
                classify_tracks(info_file_path)


# 读取 info.txt 并将各个 track 分类
def classify_tracks(info_file):
    with open(info_file, "r") as file:
        for line in file:
            # 解析每一行，提取乐器名称
            parts = line.strip().split(" - ")
            if len(parts) == 2:
                track_name = parts[0].replace("Track Name: ", "").strip()

                # 根据乐器名称分类
                if track_name in brass_instruments:
                    instrument_categories["brass"].add(track_name)
                elif track_name in woodwinds_with_reeds:
                    instrument_categories["woodwinds_with_reeds"].add(track_name)
                elif track_name in woodwinds_without_reeds:
                    instrument_categories["woodwinds_without_reeds"].add(track_name)
                elif track_name in string_instruments:
                    instrument_categories["strings"].add(track_name)


# 指定目标文件夹的路径
root_folder = "../LOP_database/aligned"

# 遍历文件夹并进行分类
traverse_and_classify_instruments(root_folder)

# 打印分类结果
for category, tracks in instrument_categories.items():
    print(f"{category.capitalize()}:")
    for track in tracks:
        print(f"  {track}")
