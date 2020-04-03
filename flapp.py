from flask import render_template , stream_with_context, request, Response , Flask

app = Flask(__name__)

import time
import os
import numpy as np
import keras
from random import randint , choice , shuffle
from keras import backend as K
import json
from json import JSONEncoder
import cv2
from firebase import firebase
from keras.utils import to_categorical

def classification_model(base , classes):

        model = keras.models.Sequential()
        model.add(base)
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dropout(0.7))
        model.add(keras.layers.Dense(64,activation='relu'))
        model.add(keras.layers.Dropout(0.7))
        model.add(keras.layers.Dense(classes,activation='softmax'))

        model.compile(optimizer= keras.optimizers.Adam(lr = 0.0001) , loss = 'categorical_crossentropy' , metrics = ['acc'])

        return model
    
class NumpyArrayEncoder(JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return JSONEncoder.default(self, obj)

@app.route('/<params>')
def index(params):
    def generate():
        start_time = time.time()
    
        firebase_app = firebase.FirebaseApplication('https://predictors-a4804.firebaseio.com', None)
        yield 'Hi '

        yield 'Initialised Firebase App '

        Usr = params.split('-')[0]
        Predictor = params.split('-')[1]

        categories = []
        vids = []

        result = firebase_app.get('/'+Usr+'/'+Predictor+'/Classes', None)

        for i in result:
            categories.append(i)
            vids_of_cat = []
            for j in result[i]:
                vids_of_cat.append(result[i][j])
            vids.append(vids_of_cat)

        training = []

        for i in range(len(categories)):
            for vid in vids[i]:

                vcap = cv2.VideoCapture(vid)

                while(True):
                    ret, frame = vcap.read()
                    if frame is not None:
                        img = cv2.resize(frame,(100,100))
                        img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
                        img = cv2.cvtColor(img , cv2.COLOR_RGB2GRAY)
                        img = img.reshape(1,100,100)
                        training.append([img , i])
                    else:
                        break

                vcap.release()
                cv2.destroyAllWindows()

        shuffle(training)

        yield 'Collected Training From Videos '

        imgs , cats = [] , []

        for f,l in training:
            imgs.append(f)
            cats.append(l)
            
        X_train = np.asarray(imgs)
        Y_train = np.asarray(cats)

        classes = len(categories)
        imgs_per_class=150

        dataset = []

        for a in range(classes):
            for image in [X_train[i] for i in range(len(X_train)) if Y_train[i] == a][:imgs_per_class]:
                dataset.append([image , a])

        shuffle(dataset)

        imgs = []
        cats = []

        for x in dataset:
            imgs.append(np.repeat(x[0][..., np.newaxis], 3, -1))
            cats.append(x[1])

        X = np.array(imgs).reshape(-1, 100, 100, 3)
        Y = to_categorical(cats , classes)

        del training
        del dataset
        del X_train
        del Y_train

        datagen = keras.preprocessing.image.ImageDataGenerator(
            rotation_range=40,
                zoom_range=0.2,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.1,
                brightness_range=(0.25, 0.75),
                horizontal_flip=True,
                fill_mode="nearest")

        datagen.fit(X)

        base = keras.models.load_model('base_model.h5')

        yield 'Base Model Loaded '

        model = classification_model(base , classes)

        model.fit_generator(datagen.flow(X , Y , batch_size = 32) , steps_per_epoch = len(X)//32 , epochs = 1, shuffle = True , verbose=0)

        yield 'Model Trained '

        model.save('model_book_bottle_ball.h5')

        P = model.get_weights()

        yield 'Model Saving In Firebase '

        numpyArrayOne = np.array(P)

        numpyData = {"array": numpyArrayOne}
        encodedNumpyData = json.dumps(numpyData, cls=NumpyArrayEncoder)

        for x in [encodedNumpyData[i:i+10000000] for i in range(0, len(encodedNumpyData), 10000000)]:
            firebase_app.post('/'+Usr+'/'+Predictor+'/Weights', x)
        
        yield 'Model Saved In Firebase '
        
        K.clear_session()
        
        yield "Execution Time = "+str(time.time() - start_time)+" Seconds "
    
    return Response(stream_with_context(generate())) 

if __name__ == "__main__":
    app.run()
