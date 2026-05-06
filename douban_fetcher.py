# -*- coding: utf-8 -*-
import requests
from bs4 import BeautifulSoup
import urllib.parse
import time
import json
import os

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

CACHE_FILE = "douban_cache.json"

# 加载本地缓存（防止重复爬虫）
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        cache = json.load(f)
else:
    cache = {}

def save_cache():
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

def fetch_douban_info(movie_name):
    if movie_name in cache:
        return cache[movie_name]

    query = urllib.parse.quote(movie_name)
    search_url = f"https://www.douban.com/search?cat=1002&q={query}"

    try:
        resp = requests.get(search_url, headers=HEADERS, timeout=5)
        soup = BeautifulSoup(resp.text, "html.parser")

        result = soup.find("div", class_="result")
        if not result:
            print(f"[WARN] 无搜索结果：{movie_name}")
            return None

        title = result.find("a", class_="nbg")["title"]
        link = result.find("a", class_="nbg")["href"]
        img = result.find("img")["src"]

        # 豆瓣详情页再抓评分
        detail_resp = requests.get(link, headers=HEADERS, timeout=5)
        detail_soup = BeautifulSoup(detail_resp.text, "html.parser")
        rating = detail_soup.find("strong", class_="ll rating_num")
        score = rating.text.strip() if rating else "暂无评分"

        info = {
            "title": title,
            "cover_url": img,
            "rating": score,
            "douban_url": link
        }

        cache[movie_name] = info
        save_cache()

        print(f"[INFO] 成功抓取：{movie_name}")
        return info

    except Exception as e:
        print(f"[ERROR] 抓取失败：{movie_name}，原因：{e}")
        return None

# 测试用例（可注释）
if __name__ == "__main__":
    test_name = "肖申克的救赎"
    info = fetch_douban_info(test_name)
    print(json.dumps(info, ensure_ascii=False, indent=2))
