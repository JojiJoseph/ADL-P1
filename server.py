
from http import HTTPStatus
from http.server import HTTPServer, SimpleHTTPRequestHandler
import json
import os
import webbrowser
import sys
import numpy as np

import tensorflow as tf

import pickle

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
# os.chdir(dname)

feature_extractor = tf.keras.applications.MobileNetV2(include_top=False)

embedding_dim = 256
units = 1024
vocab_size = 8498

from encoder import Encoder
from decoder import Decoder


encoder = Encoder(embedding_dim)
encoder(np.zeros((1, 9, 9, 1280)))
encoder.load_weights("./encoder.hdf5")
decoder = Decoder(embedding_dim,units,vocab_size)

decoder(tf.zeros((1, 1), tf.float32), tf.zeros((1,49,256), tf.float32),tf.zeros((1,units), tf.float32))
decoder.load_weights("./decoder.hdf5")
with open("./tokenizer.pkl", "rb") as f:
    tokenizer_json = pickle.load(f)
tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_json)

def evaluate_from_feature(feature, temp=1):
  caption = ""
  max_len = 30
  hidden = decoder.reset_state(1)
  x = tf.expand_dims([tokenizer.word_index["<start>"]],axis=-1)
  for i in range(max_len):
    out,hidden = decoder(x, feature,hidden)
    next_index = tf.random.categorical(out/temp, 1)[0][0].numpy()
    if next_index == tokenizer.word_index["<end>"]:
      break
    caption +=  tokenizer.index_word[next_index] + " "
    x = tf.expand_dims([next_index],axis=-1)
  return caption

def preprocess(data):
    data = np.array(data)
    # Get indices for G, B and A channels
    indices = [[i*4+3] for i in range(299*299*1)]
    # indices = np.array(indices).reshape(1, -1)
    # # Remove these unnecessary indices
    data = np.delete(data, indices).reshape(299,299,3)
    return data

def get_prediction(data, temp=1):
    image = tf.convert_to_tensor(data, tf.float32)
    image = tf.keras.applications.inception_v3.preprocess_input(image)
    image = tf.expand_dims(image, 0)
    image = feature_extractor(image)
    encoded = encoder(image)
    caption = evaluate_from_feature(encoded, temp)
    print(caption)
    return caption

class DemoServerHandler(SimpleHTTPRequestHandler):

    def send_json(self, json_message, status=HTTPStatus.OK):
        """
        Send json_message to frontend.
        json_message is a dictionary
        """
        encoded = json.dumps(json_message).encode("utf-8", "replace")
        length = len(encoded)

        self.send_response(status)
        self.send_header("Content-type", "application/json")
        self.send_header("Content-length", length)
        self.end_headers()
        self.wfile.write(encoded)

    def do_POST(self):
        length = int(self.headers.get('content-length'))
        request = json.loads(self.rfile.read(length))

        if "data" in request:
            if request["model"] == "caption":
                data = request["data"]
                temp = float(request["temp"]) or 1
                data = preprocess(data)
                prediction = get_prediction(data, temp)
                response = {"prediction": (prediction)}
                self.send_json(response)
            else:
                response = {"error":"Not Implemented"}
                self.send_json(response, HTTPStatus.NOT_IMPLEMENTED)
        else:
            response = {"error":"Not Implemented"}
            self.send_json(response, HTTPStatus.NOT_IMPLEMENTED)

def run(server_class=HTTPServer, handler_class=DemoServerHandler):
    """
    Run server on port 8000
    """
    server_address = ('', 8000)
    httpd = server_class(server_address, handler_class)
    print("Server started at port", server_address[1])
    if "open" in sys.argv:
        webbrowser.open_new_tab("http://localhost:8000")
    httpd.serve_forever()

run()
