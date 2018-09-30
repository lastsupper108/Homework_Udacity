import numpy as np 
import sys

a =10* np.random.rand(3,2,2) #big array
b = 10*np.random.rand(2,2,2) #small array

duplicates_in_b = []


sys.stdout = open("log.txt","w")

print("\na is \n",a,"\nb is \n",b)
for i,sample in enumerate(a) :
	c = b-a[i]
	print("\nc is\n",c)
	c = np.absolute(c)
	print("\nabsolute c\n",c)

	c = np.sum(np.sum(c,axis=1),axis=1)
	print("\sum c\n",c)

	ind=np.where(c > 11)[0]

	if(len(ind)>0):
		ind=np.insert(ind,0,i,axis=0)
		print("ind is \n",ind,ind.shape)
		duplicates_in_b.append(ind)

print("FinalOutput",duplicates_in_b)

def DisplayDemo():
  print(train_dataset.shape)
  random_samples = np.random.randint(len(train_dataset),size=(10))
  sub_train = train_dataset[ random_samples,: ,:] #train_dataset[0:6]
  subt_label = train_labels[ random_samples ]
  print (random_samples,sub_train.shape)
  display_images(sub_train,subt_label)


def Diffrence_array(train_dataset,test_dataset):
  duplicates_in_b = []
  for i,sample in enumerate(train_dataset) :
    diffarr = test_dataset-train_dataset[i]


    diffarr = np.sum(np.sum(np.absolute(diffarr),axis=1),axis=1)
    diffarr = diffarr/(img_size*img_size)
    print (diffarr)
    ind=np.where(diffarr < 0.1)[0]

    if(len(ind)>0):
      ind=np.insert(ind,0,i,axis=0)
      duplicates_in_b.append(ind)

  return duplicates_in_b