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
	optimizer = torch.optim.Adam(my_mps.parameters(), lr=0.000001)


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

	if test_loss_hist:
		loss_hist.append(testLoss)
		return loss_hist
	else:
		return "epochs: "+str(epochs)+" batch size: "+str(batch_size)+" bond dim: "+str(bond_dim)+" --> loss: "+str(testLoss.item())+"\n"



def loopEpochAndBondDim():
	epochs=[5]
	bondDims=[10]

	text=open("results.txt", "w")

	for epoch in epochs:
		for bond_dim in bondDims:
			text.write(str(compute(epoch, bond_dim, 50)))

	text.close()


def plotNsave(epochs, bond_dim, batch_size):

	test_hist=compute(epochs, bond_dim, batch_size, True)

	plt.plot(np.arange(epochs+1), test_hist, "k-o")
	plt.savefig("fig_bd-"+str(bond_dim)+"_bs-"+str(batch_size))

plotNsave(20, 15, 100)