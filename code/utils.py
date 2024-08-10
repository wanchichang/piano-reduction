import os
import numpy as np
import re


instrument_categories = {
    "brass": set(),
    "woodwinds_with_reeds": set(),
    "woodwinds_without_reeds": set(),
    "strings": set(),
}

instrument_categories["brass"] = {"Horn", "Trumpet", "Trombone", "Tuba"}
instrument_categories["woodwinds_with_reeds"] = {"Oboe", "Clarinet", "Bassoon"}
instrument_categories["woodwinds_without_reeds"] = {"Flute", "Piccolo"}
instrument_categories["strings"] = {"Violin", "Viola", "Violoncello", "Contrabass"}


def getPhrase(string):

    # 原始字符串
    # string = 'Matrix 43:'

    # 使用正則表達式提取數字部分
    match = re.search(r"\d+", string)
    # print(string)
    #
    # measure = string.split(": ")[1]
    # print(string)
    # 提取的數字部分
    if match:
        number = int(match.group())
        # print("Extracted number:", number)
        return number
    else:
        print("No number found in the string.")


def getTrackNumber(string):
    instrument_name = string.split(": ")[1]
    if instrument_name == "Piano":
        return 0
    if instrument_name in instrument_categories["brass"]:
        return 0
    elif instrument_name in instrument_categories["woodwinds_with_reeds"]:
        return 1
    elif instrument_name in instrument_categories["woodwinds_without_reeds"]:
        return 2
    elif instrument_name in instrument_categories["strings"]:
        return 3


# 目標資料夾路徑
target_folder = "../LOP_database/test/hand_picked_Spotify-55"


def read_pianoroll_files(folder_path, pianorollType, numOfTrack, numOfMeasure):
    print(numOfMeasure, numOfTrack)
    # 讀取 orchestra.txt
    pianoroll_file_path = os.path.join(folder_path, f"{pianorollType}.txt")
    # orchestra_file_path = os.path.join(folder_path, "orchestra.txt")
    # orchestra_data = []
    current_matrix = None
    current_track = None
    # 創建一個大小為 (364, numOfTrack, 4*64, 128) 的所有元素初始化為0的 NumPy 陣列
    data = np.zeros((numOfMeasure, numOfTrack, 4 * 16, 128))
    # 檢查陣列的形狀
    # print(data.shape)

    f = open(pianoroll_file_path, "r")
    k = f.readlines()
    # print(k[1])
    # print(k[6][0].isdigit())
    # for line in k:
    current_array = []
    cnt = 0
    measure = 0
    # line = k[10]
    # values = [int(value) for value in line.split(',')]
    # print(values)
    trackNo = -1  # prevent out of range
    # print(len(k))
    for line in k:
        # print(line)
        if line[:6] == "Phrase":
            measure = (measure + 1) % numOfMeasure
            # measure = getMeasure(line) - 1
            # print(measure)

        elif line[0].isdigit():
            # print()
            # values = line.split(",")
            values = [int(value) for value in line.split(",")]
            values = np.array(values)

            # print(values.shape)
            # nonzero_indices = np.nonzero(values)

            # print("非零值的索引：", nonzero_indices)
            current_array.append(values)
            cnt += 1
        elif line[:5] == "Track":
            trackNo = getTrackNumber(line)

            # print(measure)

        # if len(current_array) == 256:
        if cnt == 64:
            # orchestra_data.append(current_array)
            arrays_np = np.array(current_array)
            # print(arrays_np.shape)
            # print(data[measure][trackNo].shape)
            temp = data[measure][trackNo] + arrays_np
            temp[temp > 0] = 1
            data[measure][trackNo] = temp
            current_array = []
            cnt = 0
    # arrays_np = np.array(o_data)
    # print(arrays_np.shape)
    # print(len(data[0]))
    print("---------------")
    return data


def load_pianoroll_data(base_folder, folds, test_fold):
    all_data = []
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []

    for i in range(5):
        if i == test_fold:
            continue
        fold = folds[i]
        fold_data = []
        for song in fold:
            song_folder = os.path.join(base_folder, song)
            info_path = os.path.join(song_folder, "info.txt")
            # print(info_path)
            f = open(info_path, "r")
            k = f.readlines()
            print(k[0])
            numOfMeasure = getPhrase(k[0])
            # numOfMeasure = getPhrase(song)
            orchestra_data = read_pianoroll_files(
                song_folder, "orchestra", 4, numOfMeasure
            )
            piano_data = read_pianoroll_files(song_folder, "piano", 1, numOfMeasure)
            for phrase in range(len(orchestra_data)):
                X_train.append(orchestra_data[phrase])
                Y_train.append(piano_data[phrase])
    # fold_data.append((song, orchestra_data))
    # all_data.append(fold_data)
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    np.save("../LOP_database/X_train.npy", X_train)
    np.save("../LOP_database/Y_train.npy", Y_train)

    for song in folds[test_fold]:
        # print(song)
        song_folder = os.path.join(base_folder, song)
        # print(song_folder)
        info_path = os.path.join(song_folder, "info.txt")
        # print(info_path)
        f = open(info_path, "r")
        k = f.readlines()
        print(k[0])
        numOfMeasure = getPhrase(k[0])
        # print(numOfMeasure)
        #
        orchestra_data = read_pianoroll_files(song_folder, "orchestra", 4, numOfMeasure)
        piano_data = read_pianoroll_files(song_folder, "piano", 1, numOfMeasure)
        for phrase in range(len(orchestra_data)):
            X_test.append(orchestra_data[phrase])
            Y_test.append(piano_data[phrase])

    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    np.save("../LOP_database/X_test", X_test)
    np.save("../LOP_database/Y_test", Y_test)
    # print(X_test.shape, Y_test.shape)

    # return X_train, Y_train, X_test, Y_test


def read_folds(fold_folder):
    folds = []
    for fold_file in sorted(os.listdir(fold_folder)):
        if fold_file.startswith("fold_") and fold_file.endswith(".txt"):
            fold_path = os.path.join(fold_folder, fold_file)
            with open(fold_path, "r") as f:
                songs = [line.strip().split(": ")[0] for line in f]
                folds.append(songs)
    return folds


fold_folder = "../LOP_database/aligned/folds"
base_folder = "../LOP_database/aligned"

folds = read_folds(fold_folder)
load_pianoroll_data(base_folder, folds, 4)
# read_pianoroll_files("../LOP_database/test/bouliane-8/", "orchestra", 4, 6)

# X_test, Y_test = load_pianoroll_data(base_folder, folds)
# print(folds[0])
# o_data = read_pianoroll_files(
#     "../LOP_database/test/hand_picked_Spotify-55", "orchestra", 4, 364
# )
