import json
import requests

def post_data():
    
    url = 'http://localhost:5555/predict'

    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }

    data = {
        "domains": [
            {"name": "example.com"},
            {"name": "google.com"},
            {"name": "facebook.com"},
            {"name": "youtube.com"},
            {"name": "amazon.com"},
            {"name": "reddit.com"},
            {"name": "twitter.com"},
            {"name": "instagram.com"},
            {"name": "linkedin.com"},
            {"name": "netflix.com"},
            {"name": "yahoo.com"},
            {"name": "microsoft.com"},
            {"name": "ebay.com"},
            {"name": "wikipedia.org"},
            {"name": "stackoverflow.com"},
            {"name": "wordpress.com"},
            {"name": "twitch.tv"},
            {"name": "github.com"},
            {"name": "pinterest.com"},
            {"name": "imdb.com"},
            {"name": "craigslist.org"},
            {"name": "aliexpress.com"},
            {"name": "bing.com"},
            {"name": "apple.com"},
            {"name": "espn.com"},
            {"name": "bbc.co.uk"},
            {"name": "nytimes.com"},
            {"name": "walmart.com"},
            {"name": "cnn.com"},
            {"name": "instagram.com"},
            {"name": "quora.com"},
            {"name": "adobe.com"},
            {"name": "zoom.us"},
            {"name": "target.com"},
            {"name": "tiktok.com"},
            {"name": "hulu.com"},
            {"name": "salesforce.com"},
            {"name": "spotify.com"},
            {"name": "paypal.com"},
            {"name": "booking.com"},
            {"name": "yahoo.com"},
            {"name": "oracle.com"},
            {"name": "dailymotion.com"},
            {"name": "aliexpress.com"},
            {"name": "telegram.org"},
            {"name": "apartments.com"},
            {"name": "craigslist.org"},
            {"name": "nike.com"},
            {"name": "steampowered.com"},
            {"name": "spotify.com"},
            {"name": "imgur.com"},
            {"name": "wikipedia.org"},
            {"name": "imdb.com"},
            {"name": "rottentomatoes.com"},
            {"name": "twitch.tv"},
            {"name": "soundcloud.com"},
            {"name": "linkedin.com"},
            {"name": "twitter.com"},
            {"name": "stackoverflow.com"},
            {"name": "wordpress.com"},
            {"name": "github.com"},
            {"name": "behance.net"},
            {"name": "nytimes.com"},
            {"name": "walmart.com"},
            {"name": "bbc.co.uk"},
            {"name": "cnn.com"},
            {"name": "espn.com"},
            {"name": "reddit.com"},
            {"name": "facebook.com"},
            {"name": "google.com"},
            {"name": "youtube.com"},
            {"name": "amazon.com"},
            {"name": "microsoft.com"},
            {"name": "ebay.com"},
            {"name": "bing.com"},
            {"name": "apple.com"},
            {"name": "quora.com"},
            {"name": "adobe.com"},
            {"name": "zoom.us"},
            {"name": "target.com"},
            {"name": "tiktok.com"},
            {"name": "hulu.com"},
            {"name": "salesforce.com"},
            {"name": "paypal.com"},
            {"name": "booking.com"},
            {"name": "yahoo.com"},
            {"name": "oracle.com"},
            {"name": "dailymotion.com"},
            {"name": "aliexpress.com"},
            {"name": "telegram.org"},
            {"name": "apartments.com"},
            {"name": "craigslist.org"},
            {"name": "nike.com"},
            {"name": "steampowered.com"},
            {"name": "spotify.com"},
            {"name": "imgur.com"},
            {"name": "wikipedia.org"},
            {"name": "imdb.com"},
            {"name": "rottentomatoes.com"},
            {"name": "twitch.tv"},
            {"name": "soundcloud.com"},
            {"name": "linkedin.com"},
            {"name": "twitter.com"},
            {"name": "stackoverflow.com"},
            {"name": "wordpress.com"},
            {"name": "github.com"},
            {"name": "behance.net"},
            {"name": "nytimes.com"},
            {"name": "walmart.com"},
            {"name": "bbc.co.uk"},
            {"name": "cnn.com"},
            {"name": "espn.com"}
        ]
    }


    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        try:
            response_data = response.json()
            print(response_data)
        except json.JSONDecodeError as e:
            print("Invalid JSON response:", response.content)
    else:
        print("Request failed with status code:", response.status_code)

if __name__ == "__main__":
    post_data()