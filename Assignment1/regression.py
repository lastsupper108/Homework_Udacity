from __future__ import print_function
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
import cv2


def make_arrays(nb_rows, img_size):
  if nb_rows:
    dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
    labels = np.ndarray(nb_rows, dtype=np.int32)
  else:
    dataset, labels = None, None
  return dataset, labels


img_size = 28


def display_images(images,labels):
  lst = ['I','C','J','F','B','H','A','E','D','G']
  labels = [lst[i] for i in labels]
  print (labels)
  row = 3
  col = 5  
  num = len(images)

  if( len(images) > row*col ):
    print("insufficient screen space")
    return None

  fig = plt.figure(figsize=(row,col))

  for i in range(1,num+1):
    sub = fig.add_subplot(row,col,i)
    sub.set_title(labels[i-1])
    plt.imshow(images[i-1])
  plt.show()



def DisplayDemo():
  print(train_dataset.shape)
  random_samples = np.random.randint(len(train_dataset),size=(12))
  sub_train = train_dataset[ random_samples,: ,:] #train_dataset[0:6]
  subt_label = train_labels[ random_samples ]
  #print (random_samples,sub_train.shape)
  display_images(sub_train,subt_label)
  return 0


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


def ShuffleDataSetandSave(train_dataset,test_dataset,validate_dataset,train_labels,test_labels,validate_labels):
	shuff = np.arange(len(train_labels))
	np.random.shuffle(shuff)
	train_labels= train_labels[shuff]
	train_dataset=train_dataset[shuff]

	shuff = np.arange(len(test_labels))
	np.random.shuffle(shuff)
	test_labels= test_labels[shuff]
	test_dataset=test_dataset[shuff]
	validate_dataset=validate_dataset[shuff]
	validate_labels=validate_labels[shuff]

	pickle_file = os.path.join('/home/maniac/Desktop/OtherWorld/tensorflow/Udacity Course1/MyWork/Dataset', '1cleanShuffledMNIST.pickle')

	try:
		f = open(pickle_file, 'wb')
		save = {
		'train_dataset': train_dataset,
		'train_labels': train_labels,
		'valid_dataset': validate_dataset,
		'valid_labels': validate_labels,
		'test_dataset': test_dataset,
		'test_labels': test_labels,
		}
		pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
		f.close()
	except Exception as e:
		print('Unable to save data to', pickle_file, ':', e)
		raise

	statinfo = os.stat(pickle_file)
	print('Compressed pickle size:', statinfo.st_size)
	return


def fit_subset(num_samples, img_size=28):
	#flatten train test images
	X = train_dataset[0:num_samples].reshape( num_samples, img_size*img_size )
	Y = train_labels[0:num_samples]

	test = test_dataset.reshape(len(test_labels),img_size*img_size)

	# we create an instance of linear Classifier and fit the data.
	# C is inverse of regularisation strength, smaller its value larger is regularisation.
	logreg = LogisticRegression(C=1e5,solver = 'sag')
	logreg.fit( X, Y )
	score = logreg.score( test, test_labels)
	print("\nScore is : ",score)
	return logreg
# function returns trained model 

def TestModelDemo(logreg):
	#select 12 random test samples,flatten them, predict them and show 
  random_samples = np.random.randint(len(test_dataset),size=(12))
  sub_test_images = test_dataset[ random_samples,: ,:] 
  sub_test_flatten = sub_test_images.reshape(len(sub_test_images),img_size*img_size)#flatten
  prediction = logreg.predict( sub_test_flatten )
  display_images(sub_test_images,prediction)
  return 0


logreg = fit_subset(5000)
TestModelDemo(logreg)

