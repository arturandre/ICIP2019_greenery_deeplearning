import numpy as np
import os
import keras
from keras.utils import to_categorical
import imageio

# Limited to 256 classes!
def hotEncodeLabelImage(labeled_image, dictClasses, includeOthers=True):
    h, w, _ = labeled_image.shape
    indexedImage = np.zeros((h, w), dtype='uint8') #To increase the number of classes use uint16 for example
    for i in range(len(dictClasses)):
        mask = np.where(np.all(labeled_image == np.asarray(dictClasses[i][0]), axis=-1))
        indexedImage[mask] = dictClasses[i][-1]

    if includeOthers:
        return to_categorical(indexedImage, num_classes = len(dictClasses)+1) #+1 so there's an implicit 'others' class labeled as '0'
    else:
        return to_categorical(indexedImage, num_classes = len(dictClasses))

def recolorIndexedImage(indexedImage, dictClasses):
    h, w, *_ = indexedImage.shape
    labeled_image = np.zeros((h, w, 3), dtype='uint8')
    for i in range(0,len(dictClasses)):
        mask = indexedImage == np.asarray(dictClasses[i][-1])
        labeled_image[mask] = dictClasses[i][0]

    return labeled_image


#Based on: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
#Adapted for Artur A. Oliveira
class DataGenerator(keras.utils.Sequence):
    def __init__(self, list_IDs, labels, dim, gtbasefolder, batch_size=1,
             interest_classes=None, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.on_epoch_end()
        
        idx = 1
        self.dictClasses = []
        
        if interest_classes is None:
            interest_classes = []
        with open(os.path.join(gtbasefolder, 'class_dict.csv'), 'r') as class_dict:
            dataset_classes = class_dict.read()
            dataset_classes = dataset_classes.split('\n')
            if len(interest_classes) > 0:
                # [1:] -> important because first line (0) is a header
                availableClasses = [c.split(',') for c in dataset_classes[1:] if (len(c.split(',')) == 4) and c.split(',')[0] in interest_classes]
            else:
                availableClasses = [c.split(',') for c in dataset_classes[1:]]

            for c in availableClasses:
                if len(c) != 4: continue
                label, red, green, blue = c
                self.dictClasses.append([(int(red), int(green), int(blue)), idx])
                idx = idx+1
        
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            

            
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim)
        # Initialization
        X = []
        y = []

        # Generate data
        for img_filepath in list_IDs_temp:
            # Store sample
            X.append(imageio.imread(img_filepath))
  
            # Store class
            y.append(hotEncodeLabelImage(imageio.imread(self.labels[img_filepath]), self.dictClasses))
        X = np.asarray(X)
        y = np.asarray(y)

        return X, y
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))
    
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y