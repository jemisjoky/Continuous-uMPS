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

#matplotlib.use("pdf")



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
	optimizer = torch.optim.Adam(my_mps.parameters(), lr=0.1)


	data=processMnist(trainX)[:10]

	totalB=int(len(data)/batch_size)
	print(totalB)
	#data=data.transpose(0, 1)

	test_data=processMnist(testX)[:10]

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

	print("epochs: "+str(epochs)+" batch size: "+str(batch_size)+" bond dim: "+str(bond_dim)+" --> loss: "+str(testLoss)+"\n")

	return my_mps


def contractTrain(cores, values):
	
	res=cores[0]
	val=values[0]
	res=np.einsum("ijk, i->jk", res, val)
	
	for i in range(1, len(cores)):

		res=np.einsum("ik, jkm -> jim", res, cores[i])
		res=np.einsum("ijk, i->jk", res, values[i])

	return res


def contractSquareNorm(cores):

	res=np.einsum("ijk, imn ->jmkn",cores[0], cores[0])
	for i in range(1, len(cores)):

		temp=np.einsum("ijk, imn->jmkn", cores[i], cores[i])
		res=np.einsum("klij, ijmn->klmn", res, temp)

	return res


def get_QR_transform(cores):

	newCores=[]
	R=np.identity(cores[0].shape[1])

	for i in range(0, len(cores)-1):

		A=cores[i]
		newA=np.einsum("ij, kjl->kil", R, A)
		temp=newA.reshape(-1, newA.shape[-1])
		Q,R=np.linalg.qr(temp)
		Q=Q.reshape(newA.shape[0], newA.shape[1], R.shape[0])
		newCores.append(Q)

	newA=newA=np.einsum("ij, kjl->kil", R, cores[-1])	

	newCores.append(newA)
	return newCores

def square_norm_leftQR(cores):
	A=cores[-1]
	res=np.einsum("ijk,ijm->km", A, A)
	return res


def get_quasi_prob(cores, values):
	use_cores=cores[-len(values):]
	temp=contractTrain(use_cores, values)
	temp=np.einsum("ij, ik->jk", temp, temp)
	return temp


def margin(cores, given_val):

	dim=cores[0].shape[0]
	#use_cores=cores[-len(given_val)-1:]
	probs=np.zeros(dim)

	possible_values=[]

	for i in range(dim):
		curr_val=np.zeros(dim)
		curr_val[i]=1
		vals=[curr_val]
		vals+=given_val
		probs[i]=get_quasi_prob(cores, vals)
		possible_values.append(curr_val)

	return probs, possible_values


def roll(bias):

	guess=np.random.uniform()
	S=0
	for i, val in enumerate(bias):
		if guess<S+val:
			return i
		else:
			S+=val


def sample(cores, item):

	given_vals=[]

	if item==1:
		Z=square_norm_leftQR(cores).item()
		probs, vals=margin(cores, given_vals)
		res=roll(probs/Z)
		return probs[res], [vals[res]]

	else:

		p_prec, vals_prec=sample(cores, item-1)
		probs, vals=margin(cores, vals_prec)
		res=roll(probs/p_prec)
		given_vals=[vals[res]]
		given_vals+=vals_prec
		return probs[res], given_vals


def seq_to_array(seq, w, h):
	seq=np.array(seq)
	seq=seq[:, 0]	
	return seq.reshape(w, h)



#epochs, bond_dim, batch_size
bond_dim=10
my_mps = compute(10, bond_dim, 1)
cores=[]
for i in range(len(my_mps.core_tensors)):
	cores.append(my_mps.core_tensors[i].detach().numpy())
edge=my_mps.edge_vecs.detach().numpy()

firstCore=np.einsum("ijk, j->ik", cores[0], edge[0])
firstCore=firstCore.reshape(2, 1, bond_dim)
cores[0]=firstCore

endCore=np.einsum("ijk, j->ik", cores[-1], edge[1])
endCore=endCore.reshape(2, bond_dim, 1)
cores[-1]=endCore


trans_cores=get_QR_transform(cores)
Z=square_norm_leftQR(trans_cores).item()

for i in range(10):
	p, vals=sample(trans_cores, 196)
	print(p/Z)
	im=seq_to_array(vals, 14, 14)

	plt.imshow(im)
	plt.show()






