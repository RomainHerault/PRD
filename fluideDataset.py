import torch
import torch.utils.data
import cv2
import numpy as np
import os

scaleFactor = [1,np.sqrt(2),2, 2*np.sqrt(2),4]

class fluideDataset(torch.utils.data.Dataset):
    def __init__(self,root,seqLength,nbVidPerClass,train = 'train'):
        super(fluideDataset,self).__init__()
        self.datasetDir = root+'/'+train
        self.seqLength = seqLength
        self.nbVidPerClass = nbVidPerClass
        self.nbClass = len(os.listdir(root+'/'+train))//self.nbVidPerClass
        vidNames = os.listdir(root+'/'+train)
        vidNames.sort()
        self.vidNames = vidNames
    
    def getNbClass(self):
        nbClass = len(os.listdir(self.datasetDir))//self.nbVidPerClass
        return nbClass
    
    def __len__(self):
        return sum([len(os.listdir(self.datasetDir+'/'+vidName)) for vidName in self.vidNames])
        
    def __getitem__(self,idx):
        if idx<(sum([len(os.listdir(self.datasetDir+'/'+vidName)) for vidName in self.vidNames[:2]])):
            idxClass = 0
            if idx<len(os.listdir(self.datasetDir+'/'+self.vidNames[0])):
                idxVid = 0
                idxSampler = idx
            else:
                idxVid = 1
                idxSampler = idx - len(os.listdir(self.datasetDir+'/'+self.vidNames[0]))
        elif idx<sum([len(os.listdir(self.datasetDir+'/'+vidName)) for vidName in self.vidNames[:4]]):
            idxClass = 1
            if idx<sum([len(os.listdir(self.datasetDir+'/'+vidName)) for vidName in self.vidNames[:3]]):
                idxVid = 2
                idxSampler = idx - sum([len(os.listdir(self.datasetDir+'/'+vidName)) for vidName in self.vidNames[:2]])
            else:
                idxVid = 3
                idxSampler = idx - sum([len(os.listdir(self.datasetDir+'/'+vidName)) for vidName in self.vidNames[:3]])
        elif idx<sum([len(os.listdir(self.datasetDir+'/'+vidName)) for vidName in self.vidNames[:6]]):
            idxClass = 2
            if idx<sum([len(os.listdir(self.datasetDir+'/'+vidName)) for vidName in self.vidNames[:5]]):
                idxVid = 4
                idxSampler = idx - sum([len(os.listdir(self.datasetDir+'/'+vidName)) for vidName in self.vidNames[:4]])
            else:
                idxVid = 5
                idxSampler = idx - sum([len(os.listdir(self.datasetDir+'/'+vidName)) for vidName in self.vidNames[:5]])
        elif idx<sum([len(os.listdir(self.datasetDir+'/'+vidName)) for vidName in self.vidNames[:8]]):
            idxClass = 3
            if idx<sum([len(os.listdir(self.datasetDir+'/'+vidName)) for vidName in self.vidNames[:7]]):
                idxVid = 6
                idxSampler = idx - sum([len(os.listdir(self.datasetDir+'/'+vidName)) for vidName in self.vidNames[:6]])
            else:
                idxVid = 7
                idxSampler = idx - sum([len(os.listdir(self.datasetDir+'/'+vidName)) for vidName in self.vidNames[:7]])
        elif idx<sum([len(os.listdir(self.datasetDir+'/'+vidName)) for vidName in self.vidNames[:10]]):
            idxClass = 4
            if idx<sum([len(os.listdir(self.datasetDir+'/'+vidName)) for vidName in self.vidNames[:9]]):
                idxVid = 8
                idxSampler = idx - sum([len(os.listdir(self.datasetDir+'/'+vidName)) for vidName in self.vidNames[:8]])
            else:
                idxVid = 9
                idxSampler = idx - sum([len(os.listdir(self.datasetDir+'/'+vidName)) for vidName in self.vidNames[:9]])
        elif idx<sum([len(os.listdir(self.datasetDir+'/'+vidName)) for vidName in self.vidNames[:12]]):
            idxClass = 5
            if idx<sum([len(os.listdir(self.datasetDir+'/'+vidName)) for vidName in self.vidNames[:11]]):
                idxVid = 10
                idxSampler = idx - sum([len(os.listdir(self.datasetDir+'/'+vidName)) for vidName in self.vidNames[:10]])
            else:
                idxVid = 11
                idxSampler = idx - sum([len(os.listdir(self.datasetDir+'/'+vidName)) for vidName in self.vidNames[:11]])
        elif idx<sum([len(os.listdir(self.datasetDir+'/'+vidName)) for vidName in self.vidNames[:14]]):
            idxClass = 6
            if idx<sum([len(os.listdir(self.datasetDir+'/'+vidName)) for vidName in self.vidNames[:13]]):
                idxVid = 12
                idxSampler = idx - sum([len(os.listdir(self.datasetDir+'/'+vidName)) for vidName in self.vidNames[:12]])
            else:
                idxVid = 13
                idxSampler = idx - sum([len(os.listdir(self.datasetDir+'/'+vidName)) for vidName in self.vidNames[:13]])
        elif idx<sum([len(os.listdir(self.datasetDir+'/'+vidName)) for vidName in self.vidNames[:16]]):
            idxClass = 7
            if idx<sum([len(os.listdir(self.datasetDir+'/'+vidName)) for vidName in self.vidNames[:15]]):
                idxVid = 14
                idxSampler = idx - sum([len(os.listdir(self.datasetDir+'/'+vidName)) for vidName in self.vidNames[:14]])
            else:
                idxVid = 15
                idxSampler = idx - sum([len(os.listdir(self.datasetDir+'/'+vidName)) for vidName in self.vidNames[:15]])
        else:
            idxClass = 8
            if idx<sum([len(os.listdir(self.datasetDir+'/'+vidName)) for vidName in self.vidNames[:17]]):
                idxVid = 16
                idxSampler = idx - sum([len(os.listdir(self.datasetDir+'/'+vidName)) for vidName in self.vidNames[:16]])
            else:
                idxVid = 17
                idxSampler = idx - sum([len(os.listdir(self.datasetDir+'/'+vidName)) for vidName in self.vidNames[:17]])
        
        path2Clip = self.datasetDir+'/'+self.vidNames[idxVid]+'/'+str(idxSampler)+'/'
        i = 0
        clip = np.empty((0,96,256,5))
        while(i<self.seqLength):
            image = cv2.imread(path2Clip+str(i)+'.png',0).astype(np.float64)/255.0
            image = cv2.resize(image, (256,96),interpolation=cv2.INTER_AREA)
            dogImg = self.calcDoG(image)
            dogImg = np.expand_dims(dogImg,axis = 0)
            clip = np.concatenate((clip,dogImg),axis = 0)
            i = i+1
        clip = torch.from_numpy(np.array(clip.transpose([3,0,1,2])))
        idxClass = torch.tensor(idxClass).type(torch.int64)
        return clip,idxClass
        
    def calcDoG(self,image):
        dogImg = np.empty((image.shape[0],image.shape[1],0))
        dog1 = -cv2.GaussianBlur(image,(0,0),scaleFactor[0])+image
        dog1[dog1<0]=0
        dog1 = np.expand_dims(dog1,axis = 2)
        dogImg = dogImg = np.concatenate((dogImg,dog1),axis = 2)
        for k in range(len(scaleFactor)-1):
            blur1 = cv2.GaussianBlur(image,(0,0),scaleFactor[k])
            blur2 = cv2.GaussianBlur(image,(0,0),scaleFactor[k+1])
            tempDoG = blur1 - blur2 
            tempDoG[tempDoG<0]=0
            tempDoG = np.expand_dims(tempDoG,axis = 2)
            dogImg = np.concatenate((dogImg,tempDoG),axis = 2)
        return dogImg