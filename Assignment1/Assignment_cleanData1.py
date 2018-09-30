
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



pickle_file = "./Dataset/myMNIST.pickle"
with open(pickle_file, 'rb') as f:
  a= pickle.load(f)
#since file was pickled using keys 
train_labels = a['train_labels']
test_labels = a['test_labels']
validate_labels = a['valid_labels']
train_dataset = a['train_dataset']
test_dataset = a['test_dataset']
validate_dataset = a['valid_dataset']



def display_images(images,labels):
  lst = ['I','C','J','F','B','H','A','E','D','G']
  labels = [lst[i] for i in labels]
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

def print_progress(i,num):
  n=int(num/100)+1
  if(i%n == 0):
    print("..."+str(i/n)+"%...\n")
  return 0

def Diffrence_array(train_dataset,test_dataset):
 
  #sum of squares in diff image threshold 
  threshold = 0.001

  duplicates_in_b =[]

  for i,sample in enumerate(train_dataset) :

    print_progress(i,len(train_dataset))
    diffarr = test_dataset-train_dataset[i]


    diffarr = np.sum(np.sum(np.square(diffarr),axis=1),axis=1)
    diffarr = diffarr/(img_size*img_size)
    #print(diffarr)
    ind=np.where(diffarr < threshold)[0]
    #print (i,ind,"\n")
    if(len(ind)>0):
      ind=np.insert(ind,0,i,axis=0)
      duplicates_in_b.append(ind)

  #train_ele stores index of images in training dataset which have its duplicates in test
  #and test_ele stores array of elements corresponding to train_ele
  train_ele= [item[0] for item in duplicates_in_b]
  test_ele = [sublist[1:] for sublist in duplicates_in_b]
  
  return train_ele,test_ele




def Display_few_duplicate_samples(train_dataset,test_dataset,train_labels):
  train_ele,test_ele = Diffrence_array(train_dataset[0:5],test_dataset[0:50])
  print("\nbefore\n",train_ele,test_ele)

  train_dataset = np.delete(train_dataset,train_ele,axis=0)
  train_labels = np.delete(train_labels ,train_ele,axis=0)

  train_ele,test_ele = Diffrence_array(train_dataset[0:5],test_dataset[0:50])
  print("\ndelete after\n",train_ele,test_ele)

  if( len(train_ele) > 0):
    conc = np.concatenate((train_dataset[train_ele],test_dataset[test_ele]))
    conc_lab = np.concatenate((train_labels[train_ele],test_labels[test_ele]))
    display_images(conc,conc_lab)
  else:
    print("No more duplicates")
  return 0




def Save_cleaned_pickle(train_dataset,test_dataset,valid_dataset,train_labels,test_labels,valid_labels):
  

  train_ele,test_ele = Diffrence_array(train_dataset,test_dataset)
  print("\noverlap before cleaning with test\n",train_ele,"Total overlap",len(train_ele))
  print("\n length of train_data",len(train_dataset))
  train_dataset = np.delete(train_dataset,train_ele,axis=0)
  train_labels = np.delete(train_labels ,train_ele,axis=0)
  print("\n length of pruned train_data and labels",len(train_dataset),len(train_labels))

  train_ele,validate_ele = Diffrence_array(train_dataset,valid_dataset)
  print("\noverlap before cleaning with validate\n",train_ele,"Total overlap length",len(train_ele))
  print("\n length of train_data",len(train_dataset))
  train_dataset = np.delete(train_dataset,train_ele,axis=0)
  train_labels = np.delete(train_labels ,train_ele,axis=0)
  print("\n length of pruned train_data and labels",len(train_dataset),len(train_labels))
  print("\n\n Saving......")


  pickle_file = os.path.join('/home/maniac/Desktop/OtherWorld/tensorflow/Udacity Course1/MyWork/Dataset', '1cleanMNIST.pickle')

  try:
    f = open(pickle_file, 'wb')
    save = {
    'train_dataset': train_dataset,
    'train_labels': train_labels,
    'valid_dataset': valid_dataset,
    'valid_labels': valid_labels,
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

