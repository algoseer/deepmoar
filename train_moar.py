import numpy as np
import tensorflow as tf
import cv2
import sys
from tqdm import tqdm

#Load the dataset here
imageFolder='FGnet/Images'

faces=[]
trueage=[]
estages=[]
isTrain=[]

def map2class(age):
    if age<6:
        return [1,0,0,0]
    elif age<13:
        return [0,1,0,0]
    elif age<22:
        return [0,0,1,0]
    else:
        return [0,0,0,1]
        
def augmentImage(img):
    images=[]

    images.append(img)
    images.append(np.fliplr(img))

    return images


N=len(open(sys.argv[1]).readlines())

for line in tqdm(file(sys.argv[1]),total=N):
    line=line.rstrip().split()
    
    fname='%s/%s' %(imageFolder,line[0])
    pid = int(line[0][:3])
    

    img = cv2.imread(fname)
    img = cv2.resize(img,(50,50))/255.

    ages=map(float,line[1:])

    for im in augmentImage(img):

        faces.append(im)
        trueage.append(map2class(ages[0]))
        estages.append([map2class(a) for a in ages[1:4]])

        if pid>63:
            isTrain.append(False)
        else:   
            isTrain.append(True)

isTrain=np.array(isTrain)
faces=np.array(faces)
trueage=np.array(trueage)
estages=np.array(estages)
#Estimate a majority age for pretraining network
majage=estages.mean(axis=1)

N,nb_annots,nb_classes=estages.shape
inp_shape = (50,50,3)
blank_input = np.zeros(N)

# Define the keras model here for mixture of annotator reliabilities


import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Input, GlobalAveragePooling2D, Concatenate, Permute, Reshape, RepeatVector, Multiply, Add, GlobalAveragePooling1D, Lambda,, Flatten

face = Input(shape=inp_shape,name='face')
y = Conv2D(128,3,padding='same',activation='relu')(face)
y = MaxPooling2D(pool_size=(2,2))(y)
y = Conv2D(256,3,padding='same',activation='relu')(y)
y = MaxPooling2D(pool_size=(2,2))(y)
y = Conv2D(512,3,padding='same',activation='relu')(y)
y = MaxPooling2D(pool_size=(2,2))(y)
#face_embed = GlobalAveragePooling2D()(y)
face_embed = Flatten()(y)

#Predict the reliable label
y = Dense(nb_classes,activation='softmax',name='y_rel')(face_embed)

#Predict annotator reliabilities
rel = Dense(nb_annots, activation='softmax',name='rel')(face_embed)

#Now generate the biases

#blank_tensor = K.variable(np.zeros((100,1)))   # The dimension of this doesn't matter
#blank = Input(tensor=blank_tensor)
blank = Input(shape=(1,),name='dummy_zero')

biases=[]

for m in range(nb_annots):
    z=Dense(nb_classes,activation='softmax',name='ann%d' %m)(blank)
    z=Reshape((nb_classes,1))(z)
    z=Permute((2,1))(z)
    biases.append(z)

bias_concat= Concatenate(axis=1,name='ann_biases')(biases)

# Mix the biases with prediction according to reliabilities
y_mat = RepeatVector(nb_annots)(y)
rel_mat = RepeatVector(nb_classes)(rel)
rel_mat = Permute((2,1))(rel_mat)

y_rel = Multiply()([y_mat,rel_mat])
unrel_mat = Lambda(lambda x: 1-x, name = '1-rel')(rel_mat)
y_unrel = Multiply()([bias_concat, unrel_mat]) 

y_final = Add(name='noisy_y')([y_rel, y_unrel])

#Define a few other models that will be useful later
ann_rel = Model(inputs=[face],outputs=[rel])
unrel_model = Model(inputs=[blank], outputs=biases)

#Train a model on the oracle label first
rel_model = Model(inputs=[face],outputs=[y])
rel_model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
rel_model.fit(faces[isTrain], majage[isTrain], batch_size=64, epochs=50, validation_data=(faces[~isTrain], majage[~isTrain]))

loss, acc1= rel_model.evaluate(faces[~isTrain], trueage[~isTrain])


pred_model=Model(inputs = [face,blank], outputs=[y_final])

pred_model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
pred_model.fit([faces[isTrain],blank_input[isTrain]],estages[isTrain], batch_size=64, epochs=50, validation_data=([faces[~isTrain],blank_input[~isTrain]],estages[~isTrain]))
loss, acc2= rel_model.evaluate(faces[~isTrain], trueage[~isTrain])


print
print "Majority:",acc1*100
print "DeepMOAR:",acc2*100
