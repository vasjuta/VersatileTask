"""
Web Service for anyone to use!
Using Sanic framework
Endpoints: /hello (for handshake)
           /classify_artist
connects to ClassifierAPI - single entry point
"""

from sanic import Sanic
from sanic.response import HTTPResponse, text
from sanic.request import Request
from sanic.response import text, json, empty

import os
import sys
import inspect

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from ClassifierAPI import ClassifierAPI

app = Sanic("artist_classifier")

app.ctx.classifier = ClassifierAPI()


@app.post("/classify_artist")
def classify_handler(request: Request) -> HTTPResponse:
    input_param = request.json
    return json({"result": app.ctx.classifier.recognize_artist(input_param["painting"])})


@app.post("/hello")
def hello_handler(request: Request) -> HTTPResponse:
    print("hello")
    return json({"result": "OK"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=4040)
