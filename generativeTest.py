import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transform
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
# After running `make install` in the torchmps folder, this should work
from torchmps import ProbMPS

matplotlib.use("pdf")



def loadMnist():

	mnist_trainset = datasets.MNIST(root='../../mnist', train=True, download=False, transform=transform.ToTensor())
	mnist_testset = datasets.MNIST(root='../../mnist', train=False, download=False, transform=transform.ToTensor())
	return mnist_trainset.train_data, mnist_trainset.train_labels, mnist_testset.test_data, mnist_testset.test_labels

trainX,trainY, testX, testY=loadMnist()


def processMnist(x, pool=True):

	if pool==True:

		avg=nn.AvgPool2d(2)
		x=avg(x.float())

	temp=x.numpy()
	temp=np.where(temp>64, 1, 0)
	x=torch.tensor(temp)

	x=torch.flatten(x, start_dim=1)

	return x

print(testX.shape)

print(processMnist(testX).shape)

def compute(epochs, bond_dim, batch_size, test_loss_hist=False):
	#bond_dim = 10
	input_dim = 2
	#batch_size = 100
	sequence_len = 196
	complex_params = False

	#epochs=1
	loss_hist=[]

	my_mps = ProbMPS(sequence_len, input_dim, bond_dim, complex_params)
	optimizer = torch.optim.Adam(my_mps.parameters(), lr=0.00001)


	data=processMnist(trainX)[:]

	totalB=int(len(data)/batch_size)
	print(totalB)
	#data=data.transpose(0, 1)

	test_data=processMnist(testX)[:]

	def testLoop(dataTest):

		totalBT=int(len(dataTest)/batch_size)
		testLoss=torch.tensor(0.0)
		for j in range(totalBT):

			print("test       ", j)
			batchTest=dataTest[j*batch_size:min((j+1)*batch_size, len(dataTest))]
			testLoss+=my_mps.loss(batchTest.transpose(0, 1))*len(batchTest)

		testLoss=testLoss/len(dataTest)

		return testLoss.item()



	for e in range(epochs):

		if test_loss_hist:
			loss_hist.append(testLoop(test_data))

		print(e)

		for j in range(totalB):

			print("       ", j)
			batchData=data[j*batch_size:min((j+1)*batch_size, len(data))]
			
			batchData=batchData.transpose(0,1)
		# Verify that backprop works fine, and that gradients are populated
			loss = my_mps.loss(batchData)  # <- Negative log likelihood loss
			loss.backward()
			optimizer.step()

	testLoss=testLoop(test_data)

	print("epochs: "+str(epochs)+" batch size: "+str(batch_size)+" bond dim: "+str(bond_dim)+" --> loss: "+str(testLoss.item())+"\n")

	return my_mps

#tn=compute(1, 10, 100)
#n=4, m=6
firstCore=np.random.random([2,1, 6])
midCore=np.random.random([2, 6, 6])
endCore=np.random.random([2, 6, 1])
coreList=[firstCore, midCore, endCore]


allVals=[]
for i in range(2):
	for j in range(2):
		for k in range(2):
			allVals.append([[i, 1-i], [j, 1-j], [k, 1-k]])


#A=np.random.random([3, 4])
#Q, R=np.linalg.qr(A)
#print(R)

#print(Q @ Q.transpose())

#print(Q.shape)

#print(np.einsum("ji, ki -> jk", Q, Q))

def contractTrain(cores, values):
	
	res=cores[0]
	val=values[0]

	res=np.einsum("ijk, i->jk", res, val)
	
	for i in range(1, len(cores)):
		res=np.einsum("ik, jkm -> jim", res, cores[i])
		res=np.einsum("ijk, i->jk", res, values[i])
	#print(res.shape)
	#print(res)
	return res

#contractTrain(coreList, allVals[0])

def contractSquareNorm(cores):

	res=np.einsum("ijk, imn ->jmkn",cores[0], cores[0])
	for i in range(1, len(cores)):

		temp=np.einsum("ijk, imn->jmkn", cores[i], cores[i])

		res=np.einsum("klij, ijmn->klmn", res, temp)
		print(res)
	#print(res.shape)
	return res

#A=np.zeros([1, 1,1,1])
#for vals in allVals:
#	temp=contractTrain(coreList, vals)
#	A+=np.einsum("ij, kl-> ijkl",temp, temp)

#contractSquareNorm(coreList)

def get_QR_transform(cores):

	#first=cores[0]
	#temp=first.reshape(-1, first.shape[-1])
	#print(temp.shape)
	#Q,R=np.linalg.qr(temp)

	newCores=[]
	R=np.identity(cores[0].shape[1])

	for i in range(0, len(cores)-1):

		A=cores[i]
		print(R.shape)
		print(A.shape)
		newA=np.einsum("ij, kjl->kil", R, A)
		print(newA.shape)
		temp=newA.reshape(-1, newA.shape[-1])
		Q,R=np.linalg.qr(temp)
		print(Q.shape)
		print(R.shape)
		print(i)
		Q=Q.reshape(newA.shape[0], newA.shape[1], R.shape[0])
		newCores.append(Q)

	newA=newA=np.einsum("ij, kjl->kil", R, cores[-1])	

	newCores.append(newA)
	return newCores

def margin_leftQR(cores):
	A=cores[-1]
	res=np.einsum("ijk,ijm->km", A, A)
	return res

#newCores=get_QR_transform(coreList)
#print(contractSquareNorm(coreList))
#print(contractSquareNorm(newCores))
#print(margin_leftQR(newCores))

my_mps = ProbMPS(16, 2, 5, False)


cores=my_mps.core_tensors
edge=my_mps.edge_vecs
print(cores[0].shape)
print(edge)