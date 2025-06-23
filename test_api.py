import requests

url = "http://127.0.0.1:10000/"  # Or use your actual LAN IP if testing on another device

# Open your audio file in binary mode
with open("audio.mp3", "rb") as f:
    files = {"audio": f}
    response = requests.post(url, files=files)

# Print the response from the server
print(response.status_code)
print(response.json())
