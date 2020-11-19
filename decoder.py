import tensorflow as tf
import numpy as np
from attention import MyAttention



class Decoder(tf.keras.Model):
  def __init__(self, embedding_dim, units, vocab_size):
    super(Decoder, self).__init__()
    self.units = units
    
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(units, return_sequences=True, return_state=True, recurrent_initializer="glorot_uniform")
    self.fc1 = tf.keras.layers.Dense(units)
    self.fc2 = tf.keras.layers.Dense(vocab_size)

    self.attention = MyAttention(units)

    # self.preprocess_features = tf.keras.layers.Dense(64)
  
  def call(self, x, features, hidden):
    # print("x:", x)
    x = self.embedding(x)
    # print("xshape after embedding", x.shape)
    # print(features.shape)
    # features = tf.nn.relu(self.preprocess_features(features))
    context, attention_weights = self.attention(features, hidden)
    # features = tf.reshape(features, (features.shape[0], 1, -1))
    # features = tf.reduce_sum(features, axis=1, keepdims=True)
    # print(features.shape)
    # print(x.shape, context.shape)
    x = tf.concat([x, context], axis=2)
    x,state = self.gru(x)
    # print(state)
    x = self.fc1(x)
    x = self.fc2(x)
    # x = tf.squeeze(x, axis=1) # Uncomment if you want batches to be unseperated
    # print("xshape before return", x.shape)
    return x, state
  def reset_state(self, batch_size):
    return np.zeros((batch_size, self.units))