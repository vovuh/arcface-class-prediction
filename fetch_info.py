import os
import shutil
import time
from multiprocessing.dummy import Pool as ThreadPool
from urllib.parse import urljoin

import requests
import urllib3
from bs4 import BeautifulSoup


def new_item_dir(folder_name):
    dir_name = "data/" + folder_name
    os.makedirs(dir_name, exist_ok=True)
    return dir_name


def get_soup(url):
    try:
        http = urllib3.PoolManager()
        response = http.request("GET", url)
        return BeautifulSoup(response.data, features="html.parser")
    except:
        return


def parse_item(url):
    soup = get_soup(url)
    items = soup.select("li[id^=photo-]>a>img[class=lazyload]")
    item_dir = new_item_dir(url.split('/')[-2])
    for i in range(len(items)):
        try:
            item_url = urljoin(url, items[i].get("data-src"))
            r = requests.get(item_url, stream=True)
            if r.status_code == 200:
                r.raw.decode_content = True
                with open(os.path.join(item_dir, "%02d.jpg" % i), "wb") as ouf:
                    shutil.copyfileobj(r.raw, ouf)
        except:
            continue
    prop_names = soup.select("td[class=char_name]>span")
    prop_values = soup.select("td[class=char_value]>span")
    with open(os.path.join(item_dir, "props.txt"), "w") as ouf:
        for i in range(len(prop_names)):
            try:
                prop_name = str(prop_names[i].contents[0]).strip()
                prop_value = str(prop_values[i].contents[0]).strip()
                ouf.write(prop_name + "\n" + prop_value + "\n")
            except:
                continue


block_number = 0

# Checkpoint constant
LAST_PROCESSED_BLOCK = 101


def parse_section(url):
    global block_number
    last_page_index = 1
    soup = get_soup(url)
    pages = soup.select("div[class=nums]")
    if len(pages) > 0:
        a_tags = pages[0].find_all("a")
        if len(a_tags) > 0:
            href = a_tags[-1].get("href")
            last_page_index = int(href[href.find("=") + 1:])

    pages_urls = []
    for i in range(1, last_page_index + 1):
        pages_urls.append(url + "?PAGEN_1=%d" % i)

    pool = ThreadPool(64)
    for i in range(0, len(pages_urls), 64):
        block_number += 1
        if block_number <= LAST_PROCESSED_BLOCK:
            print('The block %d is skipped because it was processed earlier' % block_number)
            continue
        start_time = time.time()
        try:
            current_urls = pages_urls[i:min(i + 64, len(pages_urls))]
            items = []
            pool.map(lambda item: items.extend(get_soup(item).select("div[class=item-title]")), current_urls)
            pool.map(lambda item: parse_item(urljoin(url, item.find("a").get("href"))), items)
        except:
            continue
        end_time = time.time()
        print('The block %d processing took %.2f seconds' % (block_number, end_time - start_time))


if __name__ == "__main__":
    host_url = "https://ognisveta.ru/catalog/"
    main_soup = get_soup(host_url)
    sections = main_soup.select("td[class=section_info]")
    for section in sections:
        section_url = urljoin(host_url, section.find("a").get("href"))
        parse_section(section_url)
