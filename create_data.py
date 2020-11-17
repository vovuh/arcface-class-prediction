import os
import io

import pandas as pd


def read_data(property_name=None):
    path = os.path.join(os.getcwd(), 'dataset')
    classes = {}
    for cpath, dirs, files in os.walk(path):
        file_class = cpath.split('\\')[-1]
        if not (file_class in classes):
            idx = len(classes)
            classes[file_class] = idx
    names = []
    labels = []
    for cpath, dirs, files in os.walk(path):
        label = classes[cpath.split('\\')[-1]]
        if not (property_name is None):
            try:
                with io.open(os.path.join(cpath, "props.txt"), mode="r", encoding="utf-8") as inf:
                    lines = inf.readlines()
                    prop_names = list(map(str.strip, lines[0::2]))
                    prop_values = list(map(str.strip, lines[1::2]))
                    pos = prop_names.index(property_name)
                    property_value = prop_values[pos].split(", ")
                    if len(property_value) > 1:
                        continue
                    label = property_value[0]
            except:
                continue
        for file in files:
            if file.endswith(".txt"):
                continue
            names.append(os.path.join(cpath, file))
            labels.append(label)
    data = {"imgpath": names, "class": labels}
    print("The number of classes is %d " % len(names))
    return pd.DataFrame(data)
