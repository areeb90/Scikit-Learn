import requests

payload = {
    "pclass": 1, "age": 35.0, "sibsp": 1, "parch": 0,
    "fare": 53.10, "sex": "male", "embarked": "C",
    "family_size": 2, "is_alone": 0, "title": "Mr"
}

response = requests.post("http://127.0.0.1:8000/predict", json=payload)
print(response.json())