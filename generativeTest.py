from comet_ml import Experiment
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#torch.set_default_tensor_type('torch.cuda.FloatTensor')

experiment = Experiment("rpXxpS1xca85qSrX2xFjfgFsE",project_name="MPS_Mnist_1")


def loadMnist():

	mnist_trainset = datasets.MNIST(root='../mnist', train=True, download=False, transform=transform.ToTensor())
	mnist_testset = datasets.MNIST(root='../mnist', train=False, download=False, transform=transform.ToTensor())
	return mnist_trainset.train_data, mnist_trainset.train_labels, mnist_testset.test_data, mnist_testset.test_labels


def processMnist(x, pool=True, discrete=True):

	if pool==True:

		avg=nn.AvgPool2d(2)
		x=avg(x.float())


	if discrete==True:
		temp=x.numpy()
		temp=np.where(temp>64, 1, 0)
		x=torch.tensor(temp)

	x=torch.flatten(x, start_dim=1)

	return x

def embeddingFunc(vect, s, d):
	emb=np.cos(vect*np.pi/2)**(d-s)*np.sin(vect*np.pi/2)**(s-1)
	#print(emb.shape, vect.shape)
	return emb

def embedding(data, d):

	newEmbed=np.zeros([len(data),len(data[0]), d])
	for s in range(d):
		#print(newEmbed[:,:, s].shape)
		newEmbed[:,:, s]=embeddingFunc(data, s+1, d)

	return newEmbed



def roll(bias):

	guess=np.random.uniform()
	S=0
	for i, val in enumerate(bias):
		if guess<S+val:
			return i
		else:
			S+=val


def toyData(N):

	data=torch.zeros([N, 4, 4])

	for i in range(N):

		quarter=((np.random.random([2, 2])>0.5))
		bias=[0.333, 0.667]
		l=roll(bias)
		h=roll(bias)
		data[i, l*2:(l+1)*2, h*2:(h+1)*2]=torch.tensor(quarter)
		
	return data.long()
#(trainX, testX, epochs, seq_len, size, bond_dim, batch)
def compute(trainX, testX, epochs, seq_len, bond_dim, batch_size, lr=0.001, hist=False):

	lr=0.001

	order = np.array(range(10))
	np.random.shuffle(order)
	trainX[np.array(range(10))] = trainX[order]

	#bond_dim = 10
	input_dim = 2
	#batch_size = 100
	sequence_len = seq_len
	complex_params = False

	#epochs=1
	test_loss_hist=[]
	train_loss_hist=[]

	my_mps = ProbMPS(sequence_len, input_dim, bond_dim, complex_params)
	my_mps.to(device)
	optimizer = torch.optim.Adam(my_mps.parameters(), lr=lr)


	data=trainX

	totalB=int(len(data)/batch_size)
	print(totalB)
	#data=data.transpose(0, 1)

	test_data=testX

	def testLoop(dataTest):

		totalBT=int(len(dataTest)/batch_size)
		testLoss=0
		for j in range(totalBT):

			#print("test       ", j)
			batchTest=dataTest[j*batch_size:min((j+1)*batch_size, len(dataTest))]
			toLearn=torch.tensor(np.transpose(batchTest, [1, 0, 2]))
			testLoss+=my_mps.loss(toLearn.float()).detach().item()*len(batchTest)

		testLoss=testLoss/len(dataTest)

		return testLoss

	prevLoss=-np.log(len(data))

	for e in range(epochs):

		if hist:
			testl=testLoop(test_data)
			trainl=testLoop(data)
			test_loss_hist.append(testl)
			train_loss_hist.append(trainl)

			with experiment.train():
				experiment.log_metric("logLikelihood", trainl, step=e)
			with experiment.test():
				experiment.log_metric("logLikelihood", testl, step=e)

		print(e)

		#adapatation of the learning rate
		#if e>5:
		#	optimizer = torch.optim.Adam(my_mps.parameters(), lr=lr/10)

		#if e==5:
		#	optimizer = torch.optim.Adam(my_mps.parameters(), lr=lr/10)			

		for j in range(totalB):

			#print("       ", j)
			batchData=data[j*batch_size:min((j+1)*batch_size, len(data))]
			batchData=torch.tensor(np.transpose(batchData, [1, 0, 2]))

			loss = my_mps.loss(batchData.float())  # <- Negative log likelihood loss

			loss.backward()
			optimizer.step()


#	if hist:
#		test_loss_hist.append(testLoop(test_data))
#		train_loss_hist.append(testLoop(data))

	#print("epochs: "+str(epochs)+" batch size: "+str(batch_size)+" bond dim: "+str(bond_dim)+" --> loss: "+str(testLoss)+"\n")

	return my_mps, test_loss_hist, train_loss_hist


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



#epochs, bond_dim, batch_size, seq_len
def test(trainX, testX, epochs, bond_dim, seq_len, sizes):

	my_mps, t1, t2 = compute(trainX, testX, epochs, bond_dim, 10, seq_len)
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
		p, vals=sample(trans_cores, seq_len)
		print(p/Z)
		im=seq_to_array(vals, sizes, sizes)

		plt.imshow(im)
		plt.show()


#test()
#(trainX, testX, epochs, seq_len, size, bond_dim, batch)
def plot(trainX, testX, epochs, seq_len, bond_dim, batch_size, lr):
	#print(trainX.shape)

	my_mps, test_hist, train_hist = compute(trainX, testX, epochs,seq_len, bond_dim, batch_size, lr=lr, hist=True)
	x=np.arange(epochs)
	plt.plot(x, test_hist, "m")
	plt.plot(x, train_hist, "b")
	plt.savefig("graph1")




#N=1000
#n=100
#trainX=toyData(N).reshape(N, 16)
#testX=toyData(n).reshape(n, 16)
trainX,trainY, testX, testY=loadMnist()
#trainX=processMnist(trainX, discrete=True)
#print(trainX[0])
#testX=processMnist(testX, discrete=True)

#trainX=embedding(trainX, 2)
#testX=embedding(testX, 2)

def getTrainTest(trainX, testX, pool=False, discrete=True, d=2):
	trainX=processMnist(trainX, pool=pool, discrete=discrete)
	testX=processMnist(testX, pool=pool, discrete=discrete)
	
	trainX=embedding(trainX, d)
	testX=embedding(testX, d)
	return trainX, testX

trainX, testX=getTrainTest(trainX, testX, pool=True, discrete=True)

print(trainX)


bond_dim=20; seq_len=len(trainX[0]); size=int(np.sqrt(len(trainX[0]))); epochs=20;


hyper_params = {
	"bond_dim": 5,
    "sequence_length": len(trainX[0]),
    "input_dim": 2,
    "batch_size": 10,
    "epochs": 10,
    "lr_init": 0.0001
}

experiment.log_parameters(hyper_params)

#print(trainX[:1])
#print(embedding(trainX[:1], 2))

plot(trainX[:1000], testX[:1000], hyper_params["epochs"],hyper_params["sequence_length"],
	hyper_params["bond_dim"], hyper_params["batch_size"], lr=hyper_params["lr_init"])
#(trainX, testX, epochs, seq_len, size, bond_dim, batch)