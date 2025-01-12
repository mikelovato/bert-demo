# bert-demo
bert-demo for docker

### how to run this repository

```sh
docker build -t my-bert .

docker run -p 6372:6372 my-bert

curl -X POST -H "Content-Type: application/json" -d '{"text": "What is the capital in France?"}' http://localhost:6372/predict
```
### Prerequisites

- Docker installed on your machine
- Internet connection to pull the Docker image

### Project Structure

```
/bert-demo
├── Dockerfile
├── README.md
├── app.py
└── requirements.txt
```

### Dockerfile

```Dockerfile
FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "app.py"]
```

### app.py

```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    text = data['text']
    # Dummy prediction logic
    response = {'prediction': f'This is a dummy response for: {text}'}
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6372)
```

### requirements.txt

```
flask
```

### Testing the API

After running the Docker container, you can test the API using the following `curl` command:

```sh
curl -X POST -H "Content-Type: application/json" -d '{"text": "What is the capital in France?"}' http://localhost:6372/predict
```

You should receive a JSON response with a dummy prediction.
