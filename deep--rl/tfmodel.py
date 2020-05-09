from tensorflow.keras import layers
import tensorflow as tf


class Linear(layers.Layer):

  def __init__(self, units=32):
    super(Linear, self).__init__()
    self.units = units

  def build(self, input_shape):
    self.w = self.add_weight(shape=(input_shape[-1], self.units),
                             initializer='random_normal',
                             trainable=True)
    self.b = self.add_weight(shape=(self.units,),
                             initializer='random_normal',
                             trainable=True)

  def call(self, inputs):
    return tf.matmul(inputs, self.w) + self.b

class ANN1(layers.Layer):
    def __init__(self):
        super(ANN1, self).__init__()
        print("Tensorflow Model")
        self.linear1 = Linear(4000)
        #self.linear2 = Linear(300)
        self.linear3 = Linear(1)
        lr = 10e-5
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
        self.loss_fn = tf.keras.losses.MSE

    def call(self, x):
        x = self.linear1(x)
        x = tf.nn.relu(x)
        #x = self.linear2(x)
        #x = tf.nn.relu(x)
        x = self.linear3(x)
        return x

    
    def partial_fit(self, X, Y):
        with tf.GradientTape() as tape:
            logits = self(X)  # Logits for this minibatch
            # Loss value for this minibatch
            loss_value = sum(self.loss_fn(Y, logits))

        grads = tape.gradient(loss_value, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

    def predict(self, X):
        return self(X)
