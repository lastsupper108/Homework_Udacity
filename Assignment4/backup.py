from __future__ import print_function
from __future__ import division
import imageio
import matplotlib.pyplot as plt
import numpy as np
import sys
import tensorflow as tf 
from six.moves import cPickle as pickle
import random

image_size = 28
input_channels = 1
num_category = 10

pickle_file = "../Dataset/cleanShuffledMNIST.pickle"
with open(pickle_file, 'rb') as f:
	a= pickle.load(f)
#since file was pickled using keys 
train_labels = a['train_labels']
test_labels = a['test_labels']
validate_labels = a['valid_labels']
train_dataset = a['train_dataset']
test_dataset = a['test_dataset']
validate_dataset = a['valid_dataset']

lst = ['I','C','J','F','B','H','A','E','D','G']

def ReshapeAndOneHot(Dataset,labels):
	Dataset = Dataset.reshape((-1,image_size,image_size,1)).astype(np.float32)
	one_hot = np.zeros((len(labels),10))
	one_hot[np.arange(len(labels)),labels] = 1
	return Dataset,one_hot

train_dataset,train_labels = ReshapeAndOneHot(train_dataset,train_labels)
test_dataset,test_labels = ReshapeAndOneHot(test_dataset,test_labels)
validate_dataset,validate_labels = ReshapeAndOneHot(validate_dataset,validate_labels)


def Accuracy(prediction1,labels1):
	pred1=np.argmax(prediction1,1)
	label1=np.argmax(labels1,1)
	acc = 100*np.sum((pred1==label1)) / prediction1.shape[0]
	return acc

def MyConvNet():

	batchsize = 200
	graph = tf.Graph() 

	with graph.as_default():

		tf_label  = tf.placeholder(tf.float32, shape =(batchsize,num_category) )
		tf_train = tf.placeholder(tf.float32, shape =(batchsize,image_size,image_size,input_channels) )
		tf_sample_test = tf.placeholder(tf.float32, shape= (1,image_size,image_size,input_channels))

		tf_test = tf.constant(test_dataset)
		tf_validate = tf.constant(validate_dataset) 

		filter1 = tf.Variable(tf.truncated_normal([3,3,input_channels,5], stddev=0.1))
		bias1= tf.Variable(tf.ones([5]))

		filter2 = tf.Variable(tf.truncated_normal([1,1,5,10],stddev=0.1))
		bias2 = tf.Variable(tf.ones([10]))

		filter3 = tf.Variable(tf.truncated_normal([7,7,10,20],stddev=0.1))#3,3
		bias3 = tf.Variable(tf.ones([20]))
		## pool

		filter4 = tf.Variable(tf.truncated_normal([3,3,20,50],stddev=0.1))
		bias4 = tf.Variable(tf.ones([50]))
		## pool 
		##fully connected layer in disguise of convoluted layer by using valid stride and 
		##filter size same as input size
		filter5 = tf.Variable(tf.truncated_normal([7,7,50,64],stddev=0.1))
		bias5 = tf.Variable(tf.ones([64]))

		#last layer linear layer 
		Weight = tf.Variable(tf.truncated_normal([20,30],stddev=0.1))
		Bias = tf.Variable(tf.zeros([30]))

		WeightNext = tf.Variable(tf.truncated_normal([30,num_category],stddev=0.1))
		BiasNext = tf.Variable(tf.zeros([num_category]))

		def Model(train):
			layer = tf.nn.conv2d(train,filter1,[1,2,2,1],padding='SAME')
			layer = tf.nn.relu(layer + bias1)
			layer = tf.nn.conv2d(layer,filter2,[1,2,2,1],padding='SAME')
			layer = tf.nn.relu(layer + bias2)
			layer = tf.nn.conv2d(layer,filter3,[1,1,1,1],padding='VALID')
			layer = tf.nn.relu(layer + bias3)
			# layer = tf.nn.max_pool(layer,[1,2,2,1],[1,2,2,1],padding = 'SAME')#pool
			# layer = tf.nn.conv2d(layer,filter4,[1,1,1,1],padding='SAME')
			# layer = tf.nn.relu(layer + bias4)
			# layer = tf.nn.max_pool(layer,[1,2,2,1],[1,2,2,1],padding = 'SAME')
			# layer = tf.nn.conv2d(layer,filter5,[1,1,1,1],padding='VALID')
			# layer = tf.nn.relu(layer + bias5)
			shape = layer.get_shape().as_list()
			reshaped = tf.reshape(layer, [shape[0], shape[1] * shape[2] * shape[3]])
			reshaped = tf.nn.relu(tf.matmul(reshaped,Weight)+Bias)
			return (tf.matmul(reshaped,WeightNext)+BiasNext)


		out = Model(tf_train)
		#take softmax and compute loss
		loss = tf.reduce_mean(
			tf.nn.softmax_cross_entropy_with_logits(logits = out,labels =tf_label) )

		optimizer = tf.train.GradientDescentOptimizer(0.07).minimize(loss)


		train_prediction = tf.nn.softmax(out)
		test_prediction = tf.nn.softmax(Model(tf_test))
		validate_prediction = tf.nn.softmax(Model(tf_validate))
		sample_pred = tf.nn.softmax(Model(tf_sample_test))

	noSteps = 1000
	with tf.Session(graph = graph) as session:
		tf.global_variables_initializer().run()
		print("\n.......Initialized Convnet.........")

		for step in range(noSteps):

			offset = step*batchsize%(train_dataset.shape[0]-batchsize)
			#print(offset)
			batch_data = train_dataset[offset:(offset + batchsize), :]
			batch_labels = train_labels[offset:(offset + batchsize), :]
			sample = test_dataset[1:2,:]
			#print(np.argmax(batch_labels))
			feed_dict = {tf_train : batch_data, tf_label : batch_labels,tf_sample_test: sample }
			output = out.eval(session = session, feed_dict = feed_dict)
			#print(output.shape)
			_,predictions,l = session.run([optimizer,train_prediction,loss], feed_dict = feed_dict)

			if(step%100==0):
				print('\nloss at step %d is %f' % (step,l))

				acc = Accuracy(predictions,batch_labels) 
				print('Training accuracy is %lf ' % acc)


		print(test_dataset.shape)
		print('\nTest accuracy: %lf ' % Accuracy( test_prediction.eval(session=session,feed_dict=feed_dict),test_labels) )
		saver = tf.train.Saver()

		#Now, save the graph
		saver.save(session, './conv_model')
		print("imDone")
		session.close()

MyConvNet()

