import requests


def fetch_openwebcam_streams(api_key: str, limit: int) -> list[str]:
    headers = {"Authorization": f"Bearer {api_key}"}
    list_response = requests.get(
        "https://openwebcamdb.com/api/v1/webcams",
        params={"per_page": limit, "category": "events"},
        headers=headers,
    )
    slugs = [item["slug"] for item in list_response.json()["data"]][:limit]
    urls = []
    for slug in slugs:
        detail_response = requests.get(
            f"https://openwebcamdb.com/api/v1/webcams/{slug}",
            headers=headers,
        )
        stream_url = detail_response.json()["data"]["stream_url"]
        if stream_url:
            urls.append(stream_url)
    return urls
