from __future__ import division
import numpy as np
from keras import regularizers
from keras.models import Model,Sequential				
from keras.layers import AveragePooling2D,MaxPooling2D,Dropout,GlobalMaxPooling2D,GlobalAveragePooling2D
from keras.layers import Input, Dense, Conv2D,BatchNormalization, ZeroPadding2D, Flatten, Activation
from keras.utils import to_categorical
from keras.models import load_model
import matplotlib.pyplot as plt
import glob
import pickle

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo,encoding='latin1')
    return dict

def unwrap(filename):
	dict = unpickle(filename)

	batch_label = dict['batch_label']
	filenames = dict['filenames']
	data = dict['data']
	labels = dict['labels']
	print(batch_label)
	return data,labels

def PrepareDataset(num):
	if num>5 or num<1:
		print("Enter Valid Dataset size")
		return 0
	filepath = "./cifar-10/data*"
	testfile = "./cifar-10/test_batch"

	train_data = np.empty((num*10000,32,32,3),dtype='float32')
	train_labels = np.empty((num*10000),dtype='float32')
	fnames = glob.glob(filepath)
	fnames = fnames[0:num]
	
	for i,name in enumerate(fnames):
		# print("index",i*10000,i*10000+10000)
		data,labels = unwrap(name) 
		data = data.reshape((10000,3,32,32))
		data = np.rollaxis(data,2,1)
		data = np.rollaxis(data,3,2)
		train_data[i*10000:i*10000+10000]=data
		train_labels[i*10000:i*10000+10000]=labels
	test_data,test_labels = unwrap(testfile)
	test_data = test_data.reshape((10000,3,32,32))
	test_data = np.rollaxis(test_data,2,1)
	test_data = np.rollaxis(test_data,3,2)		
	return((train_data,train_labels),(test_data,test_labels))




def myModel(input_shape):
	#here shape is (32,32,3)
	X = Input(shape=input_shape)
	#We have accepted input in X input layer 
	#Now well build model for that input layer

	#conv_1 layer
	M = Conv2D(10,(3,3),strides=(1,1),padding="same",name = "conv_1")(X)
	M = BatchNormalization(axis = 3,name = "bn1")(M)#normalise across depth axis 
	M = Activation('relu')(M)
	M = MaxPooling2D((2,2),name="maxpool_1")(M)
	M = Dropout(0.5)(M)

	M = Conv2D(20,(3,3),strides=(1,1),padding="same",name = "conv_2")(M)
	M = BatchNormalization(axis = 3,name = "bn2")(M)#normalise across depth axis 
	M = Activation('relu')(M)
	M = MaxPooling2D((2,2),name="maxpool_2")(M)
	M = Dropout(0.5)(M)

	M = Conv2D(50,(3,3),strides=(1,1),padding="same",name = "conv_3")(M)
	M = BatchNormalization(axis = 3,name = "bn3")(M)#normalise across depth axis 
	M = Activation('relu')(M)
	M = MaxPooling2D((2,2),name="maxpool_3")(M)
	# M = Dropout(0.5)(M)

	M = Flatten()(M)
	M = Dense(300, activation='relu')(M)

	M = Dense(10,activation="softmax")(M)

	model = Model(inputs=X,outputs=M,name = "myModel")
	return model

def trainModel():

	(train_data,train_labels),(test_data,test_labels)=PrepareDataset(1)
	num_classes = 10 
	train_labels = to_categorical(train_labels, num_classes)
	test_labels = to_categorical(test_labels, num_classes)
	train_data = train_data/255.0
	test_data =test_data/ 255.0
	M = myModel((32,32,3))

	M.compile(loss='categorical_crossentropy',optimizer = "Adam",metrics=['accuracy'])

	his = M.fit(train_data,train_labels,batch_size=32,epochs = 35,
	            validation_data=(test_data[0:1000],test_labels[0:1000]),shuffle=True)
	M.save('myModel_2.h5')
	preds = M.evaluate(test_data[200:500],test_labels[200:500])
	### END CODE HERE ###
	print()
	print ("Loss = " + str(preds[0]))
	print ("Test Accuracy = " + str(preds[1]))

def EvalModel(nameofModel):
	M=load_model(nameofModel)
	num_classes = 10 

	(train_data,train_labels),(test_data,test_labels)=PrepareDataset(1)
	train_labels = to_categorical(train_labels, num_classes)
	test_labels = to_categorical(test_labels, num_classes)
	train_data = train_data/255.0
	test_data =test_data/ 255.0

	preds = M.evaluate(test_data[200:1500],test_labels[200:1500])
	print()
	print ("Loss = " + str(preds[0]))
	print ("Test Accuracy = " + str(preds[1]))
	
	# preds = M.predict(train_data[0:10])
	# print(np.argmax(preds,axis=1),np.argmax(train_labels[0:10],axis=1))

def SummaryModel(nameofModel):
	M=load_model(nameofModel)
	M.summary()

EvalModel('myModel_2.h5')
# SummaryModel('myModel.h5')
# SummaryModel('myModel_2.h5')


# trainModel()