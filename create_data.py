import os  # библиотека для работы с файловой системой
import io  # библиотека для работы с файлами

import pandas as pd  # библиотека для создания DataFrame


# функция для получения значение характеристики
# property_name внутри товара с путем path
def get_property_value(path, property_name):
    # оборачивание в try-except блок
    # для обработки ошибок, связанных с обработкой файла
    try:
        # открытие файла в кодировке utf-8 для чтения
        with io.open(os.path.join(path, "props.txt"), mode="r", encoding="utf-8") as inf:
            # получение списка строк файла
            lines = inf.readlines()
            # четные строки - названия характеристик
            prop_names = list(map(str.strip, lines[0::2]))
            # нечетные строки - значения характеристик
            prop_values = list(map(str.strip, lines[1::2]))
            # попытка найти нужную характеристику
            pos = prop_names.index(property_name)
            # получение значения этой характеристики
            # по соответствующей позиции
            property_value = prop_values[pos].split(", ")
            # в связи с тем, что, например, стилей может быть несколько,
            # было принято решение обрабатывать только те, у которых
            # указан ровно один стиль
            if len(property_value) == 1:
                # даже когда указан один стиль, в конце его записи
                # могла стоять запятая, которую нужно удалить
                return property_value[0].replace(",", "")
            else:
                # иначе возвращается None
                return None
    except:
        # при ошибках обработки тоже возвращается None
        return None


# функция для чтения данных с опциональным параметром
# property_name (=None если данные берутся для ArcFace)
def read_data(property_name=None):
    # получение пути к директории с данными
    path = os.path.join(os.getcwd(), 'dataset')
    # словарь, содержащий классы и их перенумерацию
    classes = {}

    # вспомогательная функция для добавление класса
    def add_file_class(file_class):
        # если класса еще нет в словаре
        if not (file_class in classes):
            # берется текущий размер словаря
            idx = len(classes)
            # и значением класса устанавливается этот самый размер
            # (таким образом, первый класс, попавший в обработку,
            # получит номер 0, второй - 1, и так далее)
            classes[file_class] = idx

    # перебор всех товаров внутри директории с данными
    # для получения списка всех классов
    for cpath, dirs, files in os.walk(path):
        # если никаких файлов нет, значит это директория
        # dataset/ и ее обрабатывать не нужно
        if len(files) == 0:
            continue
        # если property_name не указан
        if property_name is None:
            # то названием класса является название директории
            # и класс всегда добавляется в словарь
            # (так как он всегда существует)
            add_file_class(cpath.split('\\')[-1])
        else:
            # иначе же класс - это значение
            # соответствующей характериситики
            file_class = get_property_value(cpath, property_name)
            # если это не None, то класс добавляется в словарь
            if not (file_class is None):
                add_file_class(file_class)
    # создание списков с путями к изображениям
    names = []
    # и с их перенумерованными классами
    labels = []
    # а также подсчет количества изображений для каждого класса
    classes_count = {}
    # перебор всех товаров внутри директории с данными
    for cpath, dirs, files in os.walk(path):
        # если у заданного товара нет нужной нам характеристики,
        # то его следует пропустить
        if not (property_name is None) and get_property_value(cpath, property_name) is None:
            continue
        # обработка файлов внутри директории товара
        for file in files:
            # если это файл props.txt, то
            # он его обработка нам не нужна
            if file.endswith(".txt"):
                continue
            # добавление пути к изображению в список путей
            names.append(os.path.join(cpath, file))
            # если property_name не указано
            if property_name is None:
                # то классом изображения является
                # название директории (из которого получается
                # перенумерованный класс)
                label = classes[cpath.split('\\')[-1]]
                labels.append(label)
            else:
                # иначе же классом вялется значение
                # заданной характеристики
                label = classes[get_property_value(cpath, property_name)]
                labels.append(label)
            # если данный класс уже был в словаре
            if label in classes_count:
                # количество изображений с ним
                # увеличивается на 1
                classes_count[label] += 1
            else:
                # иначе оно присваивается 1
                classes_count[label] = 1
    # создание словаря final_classes, который опять сделает
    # перенумерацию классов для классификаторов
    final_classes = {}
    # перебор текущего списка классов
    for current_class in classes_count.keys():
        # если происходит обработка данных для ArcFace
        # ИЛИ в классе есть хотя бы 150 изображений
        if property_name is None or classes_count[current_class] >= 150:
            # то класс записывается в список финальных классов
            index = len(final_classes)
            final_classes[current_class] = index
    # списки финальных имен и классов
    final_names = []
    final_labels = []
    for i in range(len(names)):
        # если класс текущего изображения есть в final_classes,
        # то изображение добавляется в список финальных данных
        if labels[i] in final_classes:
            final_names.append(names[i])
            final_labels.append(final_classes[labels[i]])
    # создается словарь, содержащий в себе
    # список изображений и список их класов
    data = {"imgpath": final_names, "class": final_labels}
    # вывод статистической информации
    print("The number of samples is %d and the number of classes is %d" % (len(final_names), len(final_classes)))
    # создание и возвращение DataFrame по получившемуся словарю
    return pd.DataFrame(data)
