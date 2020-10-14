
import pandas as pd
import numpy as np
import tensorflow as tf

df = pd.read_csv('data/survey_results_public.csv',
		usecols=['Hobby', 'Employment', 'IDE'], dtype='category')

df = pd.get_dummies(df, prefix=['Hobby_', 'emp_'], columns=['Hobby', 'Employment'])
df['IDE'] = df['IDE'].apply(lambda s: 1 if 'Vim' in s else 0)

print(df.columns)
print(df.head())

df = df.dropna()

mask = np.random.rand(len(df)) < 0.8

train = df[mask]
test = df[~mask]

x_train = train.iloc[:,1:].values
y_train = train.iloc[:,0].values.reshape(-1,1)

x_test = test.iloc[:,1:].values
y_test = test.iloc[:,0].values.reshape(-1,1)

# % of developers who use your preferred IDE
# in the training and testing datasets
print("percentage of developers using preferred IDE in trainset: ", (len(train[train["IDE"] == 1]) / len(train)) * 100)
print("percentage of developers using preferred IDE in testset: ", (len(test[test["IDE"] == 1]) / len(test)) * 100)

learning_rate = 0.05
training_epochs = 20

tf.reset_default_graph()

# By aving 2 features: hours slept & hours studied
X = tf.placeholder(tf.float32, [None, 8], name="X")
Y = tf.placeholder(tf.float32, [None, 1], name="Y")

# Initialize our weigts & bias
W = tf.get_variable("W", [8, 1], initializer = tf.contrib.layers.xavier_initializer())
b = tf.get_variable("b", [1], initializer = tf.zeros_initializer())

Z = tf.add(tf.matmul(X, W), b)
prediction = tf.nn.sigmoid(Z)

# Calculate the cost
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Z, labels = Y))

# Use Adam as optimization method
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

cost_history = np.empty(shape=[1],dtype=float)
losses = []
train_acc = []
test_acc = []

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(training_epochs):
        _, c = sess.run([optimizer, cost], feed_dict={X: x_train, Y: y_train})
        print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
               "W=", sess.run(W), "b=", sess.run(b))
        cost_history = np.append(cost_history, c)
        losses.append(c)
        # Calculate the correct predictions
        correct_prediction = tf.to_float(tf.greater(prediction, 0.5))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.to_float(tf.equal(Y, correct_prediction)))
        print ("Train Accuracy:", accuracy.eval({X: x_train, Y: y_train}))
        train_acc.append(accuracy.eval({X: x_train, Y: y_train}))
        test_acc.append(accuracy.eval({X: x_test, Y: y_test}))
        print ("Test Accuracy:", accuracy.eval({X: x_test, Y: y_test}))
