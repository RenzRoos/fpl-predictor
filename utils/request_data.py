import requests

def request_data(url: str):
    while True:
        try:
            data = requests.get(f"{url}").json()
            return data
        except Exception as e:
            print(f"Error requesting data from {url}: {e}. Retrying...")