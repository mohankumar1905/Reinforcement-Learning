import numpy as np 
import tensorflow as tf 
import qlearning
import sys

class SGDRegessor:
    def __init__(self, D):
        print("Hello Tensorflow")
        lr = 10e-2
        
        #create inputs, targets and parameters. parameter (W) should not be one dimensional.
        self.w = tf.Variable(tf.random.normal(shape=(D, 1)), name='w')
        self.X = tf.placeholder(tf.float32, shape=(None, D), name='X')
        self.Y = tf.placeholder(tf.float32, shape=(None, ), name='Y')

        #make prediction and calculate cost.

        Y_hat = tf.reshape(tf.matmul(self.X, self.w), [-1])
        delta = self.Y - Y_hat
        cost = tf.reduce_sum(delta * delta)

        self.train_operation = tf.train.GradientDescentOptimizer(lr).minimize(cost)
        self.predict_operation = Y_hat

        #start the session and initialize parameters
        init == tf.global_variables_initializer()
        self.session = tf.InteractiveSession()
        self.session.run(init)

    def partial_fit(self, X, Y):
        self.session.run(self.train_operation, feed_dict = {self.X: X, self.Y:Y})

    def predict(self, X):
        return self.session.run(self.predict_operation, feed_dict = {self.X: X})

    
if __name__ == "__main__":
    qlearning.SGDRegessor = SGDRegessor
    if "monitor" in sys.argv:
        qlearning.main(monitor = True)
    else:
        qlearning.main()