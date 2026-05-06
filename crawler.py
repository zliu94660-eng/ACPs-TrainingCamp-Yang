import requests
from bs4 import BeautifulSoup

def get_douban_info(movie_name):
    search_url = f"https://movie.douban.com/subject_search?search_text={movie_name}&cat=1002"
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    r = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(r.text, 'html.parser')

    item = soup.select_one(".result-list .item-root")
    if not item:
        return {"cover_url": "", "rating": "无评分", "detail_url": ""}

    detail_url = item.select_one(".title a")["href"]

    detail_resp = requests.get(detail_url, headers=headers)
    detail_soup = BeautifulSoup(detail_resp.text, 'html.parser')

    cover_tag = detail_soup.select_one("#mainpic img")
    cover_url = cover_tag["src"] if cover_tag else ""

    rating_tag = detail_soup.select_one("strong.rating_num")
    rating = rating_tag.text.strip() if rating_tag else "无评分"

    return {
        "cover_url": cover_url,
        "rating": rating,
        "detail_url": detail_url
    }
