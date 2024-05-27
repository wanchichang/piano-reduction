import os


def get_unique_instruments(root_folder):
    unique_instruments = set()

    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)

        if os.path.isdir(folder_path):
            info_file_path = os.path.join(folder_path, "info.txt")

            if os.path.exists(info_file_path):
                with open(info_file_path, "r") as file:
                    for line in file:
                        if line.startswith("Track Name:"):
                            parts = line.strip().split(" - ")
                            if len(parts) == 2:
                                track_name = (
                                    parts[0].replace("Track Name: ", "").strip()
                                )
                                unique_instruments.add(track_name)

    return unique_instruments


# 指定目標資料夾的路徑
root_folder = "../LOP_database/aligned"

# 獲取所有出現過的樂器
unique_instruments = get_unique_instruments(root_folder)

# 打印所有出現過的樂器
print(len(unique_instruments))
print("Unique instruments found:")
for instrument in sorted(unique_instruments):
    print(instrument)
