import os
import io

import pandas as pd


def get_property_value(path, property_name):
    try:
        with io.open(os.path.join(path, "props.txt"), mode="r", encoding="utf-8") as inf:
            lines = inf.readlines()
            prop_names = list(map(str.strip, lines[0::2]))
            prop_values = list(map(str.strip, lines[1::2]))
            pos = prop_names.index(property_name)
            property_value = prop_values[pos].split(", ")
            if len(property_value) == 1:
                return property_value[0]
            else:
                return None
    except:
        return None


def read_data(property_name=None):
    path = os.path.join(os.getcwd(), 'dataset')
    classes = {}

    def add_file_class(file_class):
        if not (file_class in classes):
            idx = len(classes)
            classes[file_class] = idx

    for cpath, dirs, files in os.walk(path):
        if len(files) == 0:
            continue
        if property_name is None:
            add_file_class(cpath.split('\\')[-1])
        else:
            file_class = get_property_value(cpath, property_name)
            if not (file_class is None):
                add_file_class(file_class)
    names = []
    labels = []
    for cpath, dirs, files in os.walk(path):
        if property_name is None or get_property_value(cpath, property_name) is None:
            continue
        for file in files:
            if file.endswith(".txt"):
                continue
            names.append(os.path.join(cpath, file))
            if property_name is None:
                labels.append(classes[cpath.split('\\')[-1]])
            else:
                labels.append(classes[get_property_value(cpath, property_name)])
    data = {"imgpath": names, "class": labels}
    print("The number of classes is %d " % len(names))
    return pd.DataFrame(data)
