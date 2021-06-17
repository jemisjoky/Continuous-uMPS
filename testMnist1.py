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

# Manually set the Pytorch random seed for better reproducibility
torch.manual_seed(0)

### Experiment Parameters ###

bond_dim   = 15
batch_size = 100
num_epochs = 20
threshold  = 64     # Threshold for deciding if pixels are black vs white


def loadMnist():

    mnist_trainset = datasets.MNIST(root='./mnist', train=True, download=True, transform=transform.ToTensor())
    mnist_testset = datasets.MNIST(root='./mnist', train=False, download=True, transform=transform.ToTensor())
    return mnist_trainset.data, mnist_trainset.targets, mnist_testset.data, mnist_testset.targets

trainX, trainY, testX, testY=loadMnist()


def processMnist(x, pool=True):

    if pool==True:

        avg=nn.AvgPool2d(2)
        x=avg(x.float())

    # temp=x.numpy()
    # temp=np.where(temp>64, 1, 0)
    # x=torch.tensor(temp)
    x = (x > threshold).long()

    x=torch.flatten(x, start_dim=1)

    return x

print(testX.shape)

print(processMnist(testX).shape)

def compute(epochs, bond_dim, batch_size, lr=1e-3, test_loss_hist=False):
    input_dim = 2
    sequence_len = 196
    complex_params = False

    loss_hist=[]

    my_mps = ProbMPS(sequence_len, input_dim, bond_dim, complex_params)
    optimizer = torch.optim.Adam(my_mps.parameters(), lr=lr)

    data=processMnist(trainX)

    totalB=int(len(data)/batch_size)
    print(f"Training on {totalB} batches")
    #data=data.transpose(0, 1)

    test_data=processMnist(testX)

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

        print(f"Epoch {e}")

        for j in range(totalB):

            print("       ", j)
            batchData=data[j*batch_size:min((j+1)*batch_size, len(data))]            
            batchData=batchData.transpose(0,1)

            loss = my_mps.loss(batchData)
            # print(f"Batch {j}: {loss:.2f}")

            loss.backward()
            optimizer.step()

    testLoss=testLoop(test_data)

    if test_loss_hist:
        loss_hist.append(testLoss)
        return loss_hist
    else:
        return f"epochs: {epochs} batch size: {batch_size} bond dim: {bond_dim} --> loss: {testLoss.item()}\n"



def loopEpochAndBondDim():
    epochs=[5]
    bondDims=[10]

    text=open("results.txt", "w")

    for epoch in epochs:
        for bond_dim in bondDims:
            text.write(str(compute(epoch, bond_dim, batch_size)))

    text.close()


def plotNsave(epochs, bond_dim, batch_size):

    test_hist=compute(epochs, bond_dim, batch_size, test_loss_hist=True)

    plt.plot(np.arange(epochs+1), test_hist, "k-o")
    plt.savefig("fig_bd-"+str(bond_dim)+"_bs-"+str(batch_size))

plotNsave(num_epochs, bond_dim, batch_size)
