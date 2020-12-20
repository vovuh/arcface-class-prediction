import os  # библиотека для работы с файловой системой
import shutil  # еще одна библиотека для работы с файловой системой
import time  # библиотека для работы со  временем
from multiprocessing.dummy import Pool as ThreadPool  # библиотека для распараллеливания
from urllib.parse import urljoin  # библиотека для работы с URL

import requests  # библиотека для работы с интернет-запросами
import urllib3  # еще одна библиотека для работы с URL
from bs4 import BeautifulSoup  # библиотека для парсинга html


# функция для создания директории товара
def new_item_dir(folder_name):
    # запоминание относительного пути
    dir_name = "data/" + folder_name
    # создание директории с учетом того, что она уже была создана
    os.makedirs(dir_name, exist_ok=True)
    # возврат относительного пути к директории
    return dir_name


# функция для получения html-кода страницы
def get_soup(url):
    # оборачивание функции в try-except блок
    # для обработки всевозможных ошибок при http-запросе
    try:
        # создание http-соединения
        http = urllib3.PoolManager()
        # GET-запрос для получения кода страницы
        response = http.request("GET", url)
        # возврат обработанного html-кода
        return BeautifulSoup(response.data, features="html.parser")
    except:
        return


# функция для обработки товара
def parse_item(url):
    # получение html-кода страницы с товаром
    soup = get_soup(url)
    # получение всех тегов, внутри которых находятся
    # необходимые для загрузки изображения
    items = soup.select("li[id^=photo-]>a>img[class=lazyload]")
    # создание поддиректории для товара,
    # где название поддиректории - id из его URL (то есть для
    # https://ognisveta.ru/product/95438/ будет получено 95438)
    item_dir = new_item_dir(url.split('/')[-2])
    # сохранение всех изображений товара в созданную директорию
    for i in range(len(items)):
        # оборачивание в try-except блок
        # для обработки всевозможных ошибок при скачивании
        try:
            # в полученном теге берется атрибут
            # data-src, внутри которого находится относительная ссылка
            # на изображение, по которой получается полная ссылка
            item_url = urljoin(url, items[i].get("data-src"))
            # выполнение http-запроса
            r = requests.get(item_url, stream=True)
            # если запрос оказался успешным
            if r.status_code == 200:
                # декодируется контент запроса
                r.raw.decode_content = True
                # изображение сохраняется в бинарный файл
                # при помощи библиотеки shutil
                with open(os.path.join(item_dir, "%02d.jpg" % i), "wb") as ouf:
                    shutil.copyfileobj(r.raw, ouf)
        except:
            continue
    # получение тегов для названий и свойств характеристик
    prop_names = soup.select("td[class=char_name]>span")
    prop_values = soup.select("td[class=char_value]>span")
    # создание и запись характеристик в файл props.txt
    # внутри поддиректории товара
    with open(os.path.join(item_dir, "props.txt"), "w") as ouf:
        for i in range(len(prop_names)):
            # оборачивание в try-except блок в связи с тем,
            # что могут возникнуть ошибки при обработке
            # (например, свойство характирстики пустое)
            try:
                # берется контент, переводится в строку, затем
                # из начала и конца удаляются пробельные символы
                prop_name = str(prop_names[i].contents[0]).strip()
                prop_value = str(prop_values[i].contents[0]).strip()
                # после чего происходит запись характеристики в файл
                ouf.write(prop_name + "\n" + prop_value + "\n")
            except:
                continue


# номер текущего обрабатываемого блока
block_number = 0

# номер последнего обработанного блока
# (то есть последний блок, обработку которого
# можно пропустить)
LAST_PROCESSED_BLOCK = 101


# обработка категории
def parse_section(url):
    # взятие глобального значения текущего блока
    global block_number
    # по умолчанию последняя страница категории равна 1
    last_page_index = 1
    # получение html-кода категории
    soup = get_soup(url)
    # получение тега, содержащего в себе теги со ссылками
    # на другие страницы категории
    pages = soup.select("div[class=nums]")
    # если такой тег есть
    if len(pages) > 0:
        # то из него достаются теги <a>,
        # внутри которых находятся ссылки
        a_tags = pages[0].find_all("a")
        # если такие теги нашлись
        if len(a_tags) > 0:
            # из них берется последний, а у него
            # достается атрибут href
            href = a_tags[-1].get("href")
            # и при помощи простого парсинга получается число,
            # равное последнему номеру страницы
            last_page_index = int(href[href.find("=") + 1:])

    # запись всех страниц в список для параллельной обработки
    pages_urls = []
    for i in range(1, last_page_index + 1):
        pages_urls.append(url + "?PAGEN_1=%d" % i)

    # создание ThreadPool с 64 потоками
    pool = ThreadPool(64)
    # обработка страниц по 64 штуки
    for i in range(0, len(pages_urls), 64):
        # каждый блок - это 64 (или меньше) страницы, так что
        # к текущему номеру блока добавляется 1
        block_number += 1
        # если этот блок уже обрабатывался, то он просто пропускается
        if block_number <= LAST_PROCESSED_BLOCK:
            print('The block %d is skipped because it was processed earlier' % block_number)
            continue
        # получение времени старта обработки блока
        start_time = time.time()
        # оборачивание в try-except блок
        # во избежание любых ошибок, связанных с обработкой страниц
        try:
            # берется текущий блок страниц длины 64
            # (или меньше, если до конца категории осталось меньше страниц)
            current_urls = pages_urls[i:min(i + 64, len(pages_urls))]
            # создание списка с товарами
            items = []
            # применение параллельной функции для получения
            # url всех товаров на этих 64 страницах
            pool.map(lambda item: items.extend(get_soup(item).select("div[class=item-title]")), current_urls)
            # применение параллельной функции для обработки
            # всех товаров, полученных из функции выше
            pool.map(lambda item: parse_item(urljoin(url, item.find("a").get("href"))), items)
        except:
            continue
        # получение времени конца обработки
        end_time = time.time()
        # вывод статистической информации
        print('The block %d processing took %.2f seconds' % (block_number, end_time - start_time))


if __name__ == "__main__":
    # получение html-кода каталога
    host_url = "https://ognisveta.ru/catalog/"
    main_soup = get_soup(host_url)
    # получение списка секций
    sections = main_soup.select("td[class=section_info]")
    # последовательная обработка секций
    for section in sections:
        section_url = urljoin(host_url, section.find("a").get("href"))
        parse_section(section_url)
