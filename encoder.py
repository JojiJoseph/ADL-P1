import tensorflow as tf

class Encoder(tf.keras.Model):
  def __init__(self, embedding_dim):
    super(Encoder, self).__init__()
    self.cnn = tf.keras.layers.Conv2D(512,3, activation="relu")
    self.out = tf.keras.layers.Dense(embedding_dim)
  def call(self, x):
    x = self.cnn(x)
    x = tf.reshape(x, (x.shape[0], -1, x.shape[3]))
    x = self.out(x)
    return tf.nn.relu(x)