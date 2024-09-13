import requests

url = "http://172.22.72.169:8000/ask"  # cmd ipconfigden alındı
data = {
    "question": "Proje İzleme Sözleşme ve Yatırım Pay Sahipleri Sözleşmesi elektronik imza ile imzalanabilir mi?",
    "max_iterations": 1  
}

response = requests.post(url, json=data)

print("Status Code:", response.status_code)
print("Response JSON:", response.json())
