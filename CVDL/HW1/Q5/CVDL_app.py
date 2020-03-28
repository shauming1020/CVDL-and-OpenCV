import sys
from PyQt5.QtWidgets import QDialog, QApplication
from CVDL_Hw1 import Ui_Form    
###############################################################################
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import RandomSampler
from torch.nn import functional as F
from torch.autograd import Variable
############################ Golbal Parameters ################################
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
PIC_PATH = './picture'
MODEL_PATH = './model'
HIS_PATH = './history'

WORKERS = 0
LEARNING_RATE = 0.001
EPOCHS = 50
BATCH_SIZE = 8
PATIENCE = 64
PRINT_FREQ = 16
###############################################################################
print('Loading Cifar-10 ...')
global trainset, trainloader, testset, testloader
transform = transforms.Compose(
[transforms.ToTensor(),
 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=WORKERS)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=WORKERS)   

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.confidence = nn.Linear(84, 1)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        out = F.relu(self.fc2(x))
        pred = self.fc3(out)
        return pred

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

class AppWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.ui.hw5_1.clicked.connect(self.hw5_1)
        self.ui.hw5_2.clicked.connect(self.hw5_2)
        self.ui.hw5_3.clicked.connect(self.hw5_3)
        self.ui.hw5_4.clicked.connect(self.hw5_4)
        self.ui.hw5_5.clicked.connect(self.hw5_5)
        
        self.show()
        
    def hw5_1(self):   
        print('Deal with hw5_1 ...')
        # get some random training images
        sampler = RandomSampler(trainset, replacement=True, num_samples=10)
        images, labels = [], []
        for i in sampler:
            img, label = trainset[i]
            images.append(img.view(1,3,32,32))
            labels.append(label)
        images = torch.cat(images)
        # show images
        imshow(torchvision.utils.make_grid(images))
        # print labels
        print(' '.join('%5s' % classes[labels[j]] for j in range(10)))   
        print('')
        
    def hw5_2(self):    
        print('hyperparameters:')
        print('batch size:',BATCH_SIZE)
        print('learning rate:', LEARNING_RATE)
        print('optimizer:','Adam')
        print('')

    def hw5_3(self):
        print('Deal with hw5_3 ...')
        model = LeNet5().cuda()
        epochs_loss_history = []
        
        # define loss function (criterion) and optimizer
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE) 
        cudnn.benchmark = True
        
        for epoch in range(1):  # loop over the dataset multiple times
            epoch_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                batch_start_time = time.time()
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs, labels = Variable(inputs), Variable(labels)
        
                # zero the parameter gradients
                optimizer.zero_grad()
        
                # forward + backward + optimize
                outputs = model(inputs.cuda()).float()
                loss = criterion(outputs, labels.cuda())
                loss.backward()
                optimizer.step()
                      
                # print statistics
                epoch_loss += loss.item()
                epoch_loss += loss.item()
              
                if i % PRINT_FREQ == (PRINT_FREQ-1): # print every PRINT_FREQ mini-batches
                    print('[%d, %5d] %2.2f sec(s) loss: %.3f' %
                          (epoch + 1, i + 1, time.time()-batch_start_time,\
                           epoch_loss / PRINT_FREQ))
                    epochs_loss_history.append(epoch_loss / PRINT_FREQ)
                    epoch_loss = 0.0
                    batch_start_time = 0.0
    
        print('Finished One Epoch Training')             
        
        plt.clf()
        plt.plot(epochs_loss_history,'b')
        plt.legend(['Training'], loc="upper left")
        plt.ylabel("loss")
        plt.xlabel("Iteration")
        plt.title("Training Process")
        plt.savefig(PIC_PATH+'/'+'_hw5.3_LeNet5'+'_history.png')
        plt.show()
        plt.close()       
        print('')
    
    def hw5_4(self):
        print('Deal with hw5_4 ...')
        """
        model = LeNet5().cuda()
        loss_history, acc_history = [], []
        testing_acc_history = []  
        best_acc = 0.0
            
        # define loss function (criterion) and optimizer
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE) 
        cudnn.benchmark = True
        
        def adjust_learning_rate(optimizer, epoch):
            lr = LEARNING_RATE * (0.8 ** (epoch // 10))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        
        for epoch in range(EPOCHS):  # loop over the dataset multiple times
            epoch_start_time = time.time()
            training_loss, training_acc = 0.0, 0.0
            testing_acc = 0.0
            
            adjust_learning_rate(optimizer, epoch)
            
            model.train()
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs, labels = Variable(inputs), Variable(labels)
                
                # zero the parameter gradients
                optimizer.zero_grad()
        
                # forward + backward + optimize
                outputs = model(inputs.cuda()).float()
                loss = criterion(outputs, labels.cuda())
                loss.backward()
                optimizer.step()
                      
                # compute the loss and acc
                training_loss += loss.item()
                training_acc += np.sum(np.argmax(outputs.cpu().data.numpy(), axis=1) == labels.numpy())
                  
            model.eval()                                 
            for data in testloader:
                with torch.no_grad():
                    inputs, labels = data
                    outputs = model(inputs.cuda()).float()
                testing_acc += np.sum(np.argmax(outputs.cpu().data.numpy(), axis=1) == labels.numpy())                                          

            training_loss = training_loss/trainset.__len__()
            training_acc = training_acc/trainset.__len__() * 100
            testing_acc = testing_acc/testset.__len__() * 100
            loss_history.append(training_loss)
            acc_history.append(training_acc)
            testing_acc_history.append(testing_acc)

            print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.3f Loss: %3.3f | Testing Acc: %3.3f ' % \
                    (epoch + 1, EPOCHS, time.time()-epoch_start_time, \
                     training_acc, training_loss, testing_acc))
            epoch_start_time = 0.0
            
            # Early Stopping
            if (training_acc > best_acc):
                torch.save(model.state_dict(), MODEL_PATH+'/_hw5.4_the_best_model.pth')
                best_acc = training_acc
                PATIENCE = 0
                print ('Early Stopping, Model Saved!') 
            else:
                PATIENCE += 1             
        print('Finished Training')
  
        # Save the last model and history
        torch.save(model.state_dict(), MODEL_PATH+'/_hw5.4_the_last_model.pth')
        print ('Save the last model!')
        np.save(HIS_PATH+'/hw5.4_the_loss_history.npy',np.asarray(loss_history))
        np.save(HIS_PATH+'/hw5.4_the_acc_history.npy',np.asarray(acc_history))   
        np.save(HIS_PATH+'/hw5.4_the_testing_acc_history.npy',np.asarray(testing_acc_history))
        print ('Save the history!')
        """      

        print('Loading the history ...')
        loss_history = np.load(HIS_PATH+'/hw5.4_the_loss_history.npy')
        acc_history = np.load(HIS_PATH+'/hw5.4_the_acc_history.npy')
        testing_acc_history = np.load(HIS_PATH+'/hw5.4_the_testing_acc_history.npy')
                
        plt.clf()
        plt.plot(loss_history,'b')
        plt.legend(['Training'], loc="upper left")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.title("Training Process")
        plt.savefig(PIC_PATH+'/_hw5.4_loss_history.png')
        plt.show()
        plt.close()
        
        plt.clf()
        plt.plot(acc_history,'b')
        plt.plot(testing_acc_history,'r')    
        plt.legend(['Training', 'Testing'], loc="upper left")
        plt.ylabel("acc")        
        plt.xlabel("epoch")
        plt.title("Training Process")
        plt.savefig(PIC_PATH+'/_hw5.4_acc_history.png')
        plt.show()
        plt.close()     
        print('')
        
    def hw5_5(self):
        print('Deal with hw5_5 ...')
        index = self.ui.lineEdit.text()
        model = LeNet5().cuda()
        model.load_state_dict(torch.load(MODEL_PATH+'/_hw5.4_the_best_model.pth'))
        
        test_img = testset[int(index)][0] # (img_matrix, label) 
        imshow(torchvision.utils.make_grid(test_img))
        
        output = model(test_img.view(1,3,32,32).cuda()).data.cpu().numpy()
        
        Max, Min = np.max(output), np.min(output)
        
        output_norm = (output-Min)/(Max-Min)
        output_norm = output_norm.reshape(-1)
        
        total = np.sum(output_norm)
        output_norm /= total
        
        # the bar of the data
        plt.bar(np.arange(10), output_norm)
        plt.xticks(np.arange(10), classes)
        plt.savefig(PIC_PATH+'/_hw5.5_bar.png')
        plt.show()
        plt.close()   
        print('')

app = QApplication(sys.argv)
w = AppWindow()
w.show()
sys.exit(app.exec_())

