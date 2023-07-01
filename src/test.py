import json
import requests

def post_data():
    
    url = 'http://localhost:5555/predict'
    data = [
        {
            "domains": [
                {
                    "name": "google.com"
                },
            ]
        },
    ]

    x = requests.post(url, json=data)

    print(x.text)

if __name__ == "__main__":
    post_data()