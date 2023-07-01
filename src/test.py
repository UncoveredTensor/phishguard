import json
import requests

def post_data():
    
    url = 'localhost:5555/predict'
    data = [
        {
            "url": "https://www.google.com"
        },
    ]

    x = requests.post(url, json=data)

    print(x.text)


if __name__ == "__main__":
    post_data()