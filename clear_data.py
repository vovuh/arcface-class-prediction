import os
import shutil

from PIL import Image

if __name__ == "__main__":
    list_bad = []
    for cpath, dirs, files in os.walk(os.path.join(os.getcwd(), 'dataset')):
        for file in files:
            if file.endswith('.txt'):
                continue
            try:
                img = Image.open(os.path.join(cpath, file))
                img.verify()
            except:
                list_bad.append(cpath)
                break
    for path in list_bad:
        shutil.rmtree(path)
