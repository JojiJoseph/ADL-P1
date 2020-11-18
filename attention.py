import tensorflow as tf
class MyAttention(tf.keras.Model):
  def __init__(self, units):
    super(MyAttention, self).__init__()
    self.W11 = tf.keras.layers.Dense(units//2, activation="relu")
    self.W12 = tf.keras.layers.Dense(units)
    self.W21 = tf.keras.layers.Dense(units//2, activation="relu")
    self.W22 = tf.keras.layers.Dense(units)
    self.score = tf.keras.layers.Dense(1)

  def call(self, features, hidden):
    # print("#1", features.shape, hidden.shape)
    hidden_t = tf.expand_dims(hidden, 1)

    attention_layer = tf.nn.tanh(self.W12(self.W11(features)) + self.W22(self.W21(hidden_t)))
    score = self.score(attention_layer)
    attention_weights = tf.nn.softmax(score, axis=1)
    context_vector = tf.reduce_sum(attention_weights*features,axis=1,keepdims=True)
    return context_vector, attention_weights
