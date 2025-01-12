# bert-demo
bert-demo for docker

### how to run this repository

```sh
docker build -t my-bert .

docker run -p 6372:6372 my-bert

curl -X POST -H "Content-Type: application/json" -d '{"text": "Hello, world!"}' http://localhost:6372/predict
```
