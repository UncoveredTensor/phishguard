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
            {"name": "stackoverflow.com"},
            {"name": "domain10.com"}
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