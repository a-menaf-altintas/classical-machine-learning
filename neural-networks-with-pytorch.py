from pickletools import optimize
import torch, torchvision
import torch.optim as optim
from neural_class import *
from torchvision import transforms, datasets
import matplotlib.pyplot as plt


# import data from MINST that will be used as a training data set
test_data = datasets.MNIST('', train=False, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))

train_data = datasets.MNIST('', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))

test_data_set = torch.utils.data.DataLoader(test_data, batch_size=20, shuffle=False) # Does not need to be shuffled
train_data_set = torch.utils.data.DataLoader(train_data, batch_size=20, shuffle=True) # Needs to be shuffled to prevent biasing

n_pixel = 28 # matrix dimension is 28 x 28 in MNIST

n_epocs = 5

neural_net_obj = NeuralNet()
optimizer = optim.Adam(neural_net_obj.parameters(), lr=0.0001) # Adaptive Momentum Optimizer

for epoch in range(n_epocs): # Train data and pass n_epocs times over data
    for data_set in train_data_set:
        X, y = data_set # X: batch features, y: batch targets
        neural_net_obj.zero_grad() # You need to set gradient to zero before starting each calculation
        result = neural_net_obj(X.view(-1, n_pixel**2)) # reshape the data
        loss_value = F.nll_loss(result, y) # Compute loss value
        loss_value.backward() # apply backward the loos value
        optimizer.step() # optimize weights 
    print("Loss: ",loss_value.item()) 



# Test accuracy of the model
accuracy = 0
sum = 0

with torch.no_grad(): # Start the test without optimization
    for data_set in test_data_set:
        X, y = data_set
        result_test = neural_net_obj(X.view(-1,n_pixel**2))
 
        for index, i in enumerate(result_test):

            if torch.argmax(i) == y[index]:
                accuracy += 1
            sum += 1

print("Accuracy: ", round(accuracy/sum, 2))

plt.imshow(X[3].view(n_pixel,n_pixel))
plt.show()

# Print the test result and compare with image
print(torch.argmax(neural_net_obj(X[3].view(-1,n_pixel **2))[0]))