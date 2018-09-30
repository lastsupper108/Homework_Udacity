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

def flattenAndOneHot(Dataset,labels):
	Dataset = Dataset.reshape((-1,image_size*image_size)).astype(np.float32)
	one_hot = np.zeros((len(labels),10))
	one_hot[np.arange(len(labels)),labels] = 1
	return Dataset,one_hot

train_dataset,train_labels = flattenAndOneHot(train_dataset,train_labels)
test_dataset,test_labels = flattenAndOneHot(test_dataset,test_labels)
validate_dataset,validate_labels = flattenAndOneHot(validate_dataset,validate_labels)

num_category = 10

def Accuracy(prediction1,labels1):
	pred1=np.argmax(prediction1,1)
	label1=np.argmax(labels1,1)
	acc = 100*np.sum((pred1==label1)) / prediction1.shape[0]
	return acc

#Make True to run Stocastic gradient decent
def SGDRegularise():
	batchsize = 200

	graph = tf.Graph()
	with graph.as_default():

		tf_train_labels = tf.placeholder( tf.float32, shape =(batchsize,num_category) )
		tf_train_data = tf.placeholder( tf.float32, shape =(batchsize,image_size*image_size) )
		beta_regul = tf.placeholder(tf.float32)
		tf_validate = tf.constant(validate_dataset)
		tf_test = tf.constant(test_dataset)

		Weight  = tf.Variable( tf.truncated_normal([image_size * image_size,num_category]))
		bias = tf.Variable(tf.zeros(num_category))

		mat = tf.matmul(tf_train_data,Weight) + bias
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = mat,labels =tf_train_labels)+ beta_regul*tf.nn.l2_loss(Weight)) 
		
		optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

		train_prediction = tf.nn.softmax(mat)
		valid_prediction = tf.nn.softmax(tf.matmul(tf_validate,Weight)+bias)
		test_prediction = tf.nn.softmax(tf.matmul(tf_test,Weight)+bias)


		#lets Run the graph batchwise 
	noSteps = 900
	lossArray = np.zeros(1) #array to store and plot loss
		
	with tf.Session(graph = graph) as session:

		tf.global_variables_initializer().run()
		print("\n.......Initialized Stocastic Gradient Decent.........")

		for step in range(noSteps):
			#random offset value less than dataset by batchsize
			offset = random.randint(1,train_dataset.shape[0]-batchsize)
			#print(offset)
			batch_data = train_dataset[offset:(offset + batchsize), :]
			batch_labels = train_labels[offset:(offset + batchsize), :]
		    # Prepare a dictionary telling the session where to feed the minibatch.		
		    # The key of the dictionary is the placeholder node of the graph to be fed,
		    # and the value is the numpy array to feed to it.
			feed_dict = {tf_train_data : batch_data, tf_train_labels : batch_labels,beta_regul : 1e-3}
			_,predictions,l = session.run([optimizer,train_prediction,loss], feed_dict = feed_dict)

			if(step%100==0):
				print('\nloss at step %d is %f' % (step,l))
				lossArray = np.concatenate((lossArray, [l]))
				acc = Accuracy(predictions,batch_labels) 
				print('Training accuracy is %lf ' % acc)
				accv = Accuracy(valid_prediction.eval(session=session),validate_labels)
				print('Validation accuracy is %lf '% accv)
				
			# Calling .eval() on valid_prediction is basically like calling run(), but
			# just to get that one numpy array. Note that it recomputes all its graph
			# dependencies.

		x = np.arange(300, 1000, 100)
		plt.plot(x, lossArray[3:], linewidth=2)
		# plt.yscale('log')
		plt.show()
		print('\nTest accuracy: %lf ' % Accuracy( test_prediction.eval(session=session),test_labels) )
		session.close()


def SGDReluRegulariseDemo():
	batchsize = 200
	middle_layerSize = 1024

	graph = tf.Graph()
	with graph.as_default():

		tf_train_labels = tf.placeholder( tf.float32, shape =(batchsize,num_category) )
		tf_train_data = tf.placeholder( tf.float32, shape =(batchsize,image_size*image_size) )
		beta_regul = tf.placeholder(tf.float32)
		tf_validate = tf.constant(validate_dataset)
		tf_test = tf.constant(test_dataset)

		Weight  = tf.Variable( tf.truncated_normal([image_size * image_size,middle_layerSize]))
		bias = tf.Variable(tf.zeros(middle_layerSize))

		WeightNext = tf.Variable(tf.truncated_normal([middle_layerSize,num_category]))
		biasNext = tf.Variable(tf.zeros(num_category))

		layer= tf.nn.relu(tf.matmul(tf_train_data,Weight)+bias)

		mat = tf.matmul(layer,WeightNext) + biasNext
		
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = mat,labels =tf_train_labels) + tf.nn.l2_loss(beta_regul*Weight)+ tf.nn.l2_loss(beta_regul*WeightNext))
		
		optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

		train_prediction = tf.nn.softmax(mat)
		layer_valid = tf.nn.relu(tf.matmul( tf_validate,Weight)+bias )
		valid_prediction = tf.nn.softmax( tf.matmul(layer_valid,WeightNext)+biasNext )
		layer_test = tf.nn.relu(tf.matmul( tf_test,Weight)+bias )
		test_prediction = tf.nn.softmax( tf.matmul(layer_test,WeightNext)+biasNext  )

		#lets Run the graph batchwise 
	noSteps = 800
		
	with tf.Session(graph = graph) as session:

		tf.global_variables_initializer().run()
		print("\n.......Initialized SGD with Relu.........")

		for step in range(noSteps):
			#random offset value less than dataset by batchsize
			offset = random.randint(1,train_dataset.shape[0]-batchsize)
			#print(offset)
			batch_data = train_dataset[offset:(offset + batchsize), :]
			batch_labels = train_labels[offset:(offset + batchsize), :]
		    # Prepare a dictionary telling the session where to feed the minibatch.		
		    # The key of the dictionary is the placeholder node of the graph to be fed,
		    # and the value is the numpy array to feed to it.
			feed_dict = {tf_train_data : batch_data, tf_train_labels : batch_labels, beta_regul:1e-3}
			_,predictions,l = session.run([optimizer,train_prediction,loss], feed_dict = feed_dict)

			if(step%100==0):
				print('\nloss at step %d is %f' % (step,l))

				acc = Accuracy(predictions,batch_labels) 
				print('Training accuracy is %lf ' % acc)
				acc = Accuracy(valid_prediction.eval(session=session),validate_labels)
				print('Validation accuracy is %lf '% acc)
			# Calling .eval() on valid_prediction is basically like calling run(), but
			# just to get that one numpy array. Note that it recomputes all its graph
			# dependencies.

		print('\nTest accuracy: %lf ' % Accuracy( test_prediction.eval(session=session),test_labels) )
		session.close()



def Dropout():
	batchsize = 200
	middle_layerSize = 1024

	graph = tf.Graph()
	with graph.as_default():

		tf_train_labels = tf.placeholder( tf.float32, shape =(batchsize,num_category) )
		tf_train_data = tf.placeholder( tf.float32, shape =(batchsize,image_size*image_size) )
		
		tf_validate = tf.constant(validate_dataset)
		tf_test = tf.constant(test_dataset)

		Weight  = tf.Variable( tf.truncated_normal([image_size * image_size,middle_layerSize]))
		bias = tf.Variable(tf.zeros(middle_layerSize))

		WeightNext = tf.Variable(tf.truncated_normal([middle_layerSize,num_category]))
		biasNext = tf.Variable(tf.zeros(num_category))

		layer= tf.nn.relu(tf.matmul(tf_train_data,Weight)+bias)
		drop = tf.nn.dropout(layer,0.5) # we have to give probibility of keeping a value(1/prob. of squashing)
		mat = tf.matmul(drop,WeightNext) + biasNext
		
		loss = tf.reduce_mean(
			tf.nn.softmax_cross_entropy_with_logits(logits = mat,labels =tf_train_labels) )
		
		optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

		train_prediction = tf.nn.softmax(mat)
		layer_valid = tf.nn.relu(tf.matmul( tf_validate,Weight)+bias )
		valid_prediction = tf.nn.softmax( tf.matmul(layer_valid,WeightNext)+biasNext )
		layer_test = tf.nn.relu(tf.matmul( tf_test,Weight)+bias )
		test_prediction = tf.nn.softmax( tf.matmul(layer_test,WeightNext)+biasNext  )

		#lets Run the graph batchwise 
	noSteps = 5000
		
	with tf.Session(graph = graph) as session:

		tf.global_variables_initializer().run()
		print("\n.......Initialized SGD with Relu.........")

		for step in range(noSteps):
			#use random offset value less than dataset by batchsize
			#offset = random.randint(1,train_dataset.shape[0]-batchsize)
			# we can also use over whole data shifted offest
			offset = step*batchsize%(train_dataset.shape[0]-batchsize)
			#print(offset)
			batch_data = train_dataset[offset:(offset + batchsize), :]
			batch_labels = train_labels[offset:(offset + batchsize), :]
		    # Prepare a dictionary telling the session where to feed the minibatch.		
		    # The key of the dictionary is the placeholder node of the graph to be fed,
		    # and the value is the numpy array to feed to it.
			feed_dict = {tf_train_data : batch_data, tf_train_labels : batch_labels}
			_,predictions,l = session.run([optimizer,train_prediction,loss], feed_dict = feed_dict)

			if(step%100==0):
				print('\nloss at step %d is %f' % (step,l))

				acc = Accuracy(predictions,batch_labels) 
				print('Training accuracy is %lf ' % acc)
				acc = Accuracy(valid_prediction.eval(session=session),validate_labels)
				print('Validation accuracy is %lf '% acc)
			# Calling .eval() on valid_prediction is basically like calling run(), but
			# just to get that one numpy array. Note that it recomputes all its graph
			# dependencies.

		print('\nTest accuracy: %lf ' % Accuracy( test_prediction.eval(session=session),test_labels) )
		session.close()

Dropout()


# Implement Deeper Network with exponential decay in learning rate 
def DeepDropout():
	batchsize = 200
	middle_layerSize = 1024

	graph = tf.Graph()
	with graph.as_default():

		tf_train_labels = tf.placeholder( tf.float32, shape =(batchsize,num_category) )
		tf_train_data = tf.placeholder( tf.float32, shape =(batchsize,image_size*image_size) )
		
		tf_validate = tf.constant(validate_dataset)
		tf_test = tf.constant(test_dataset)

		Weight  = tf.Variable( tf.truncated_normal([image_size * image_size,middle_layerSize]))
		bias = tf.Variable(tf.zeros(middle_layerSize))

		WeightNext = tf.Variable(tf.truncated_normal([middle_layerSize,num_category]))
		biasNext = tf.Variable(tf.zeros(num_category))

		layer= tf.nn.relu(tf.matmul(tf_train_data,Weight)+bias)
		drop = tf.nn.dropout(layer,0.5) # we have to give probibility of keeping a value(1/prob. of squashing)
		mat = tf.matmul(drop,WeightNext) + biasNext
		
		loss = tf.reduce_mean(
			tf.nn.softmax_cross_entropy_with_logits(logits = mat,labels =tf_train_labels) )
		
		optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

		train_prediction = tf.nn.softmax(mat)
		layer_valid = tf.nn.relu(tf.matmul( tf_validate,Weight)+bias )
		valid_prediction = tf.nn.softmax( tf.matmul(layer_valid,WeightNext)+biasNext )
		layer_test = tf.nn.relu(tf.matmul( tf_test,Weight)+bias )
		test_prediction = tf.nn.softmax( tf.matmul(layer_test,WeightNext)+biasNext  )

		#lets Run the graph batchwise 
	noSteps = 5000
		
	with tf.Session(graph = graph) as session:

		tf.global_variables_initializer().run()
		print("\n.......Initialized SGD with Relu.........")

		for step in range(noSteps):
			#use random offset value less than dataset by batchsize
			#offset = random.randint(1,train_dataset.shape[0]-batchsize)
			# we can also use over whole data shifted offest
			offset = step*batchsize%(train_dataset.shape[0]-batchsize)
			#print(offset)
			batch_data = train_dataset[offset:(offset + batchsize), :]
			batch_labels = train_labels[offset:(offset + batchsize), :]
		    # Prepare a dictionary telling the session where to feed the minibatch.		
		    # The key of the dictionary is the placeholder node of the graph to be fed,
		    # and the value is the numpy array to feed to it.
			feed_dict = {tf_train_data : batch_data, tf_train_labels : batch_labels}
			_,predictions,l = session.run([optimizer,train_prediction,loss], feed_dict = feed_dict)

			if(step%100==0):
				print('\nloss at step %d is %f' % (step,l))

				acc = Accuracy(predictions,batch_labels) 
				print('Training accuracy is %lf ' % acc)
				acc = Accuracy(valid_prediction.eval(session=session),validate_labels)
				print('Validation accuracy is %lf '% acc)
			# Calling .eval() on valid_prediction is basically like calling run(), but
			# just to get that one numpy array. Note that it recomputes all its graph
			# dependencies.

		print('\nTest accuracy: %lf ' % Accuracy( test_prediction.eval(session=session),test_labels) )
		session.close()
