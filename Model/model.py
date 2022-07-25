import string
from Utils import params as p
import numpy as np
from tensorflow import keras
import numpy as np
print(f"Numpy: {np.version.version}")



model=keras.models.load_model('C://Users//user1//Documents//bootcamp//Project//keras_cifar10_trained_model.h5')

def predict(img:np.ndarray)->string:
    img = img.astype('float32')
    img /= 255
    img=img.reshape(-1,img.shape[0],img.shape[1],img.shape[2])
    # label=model(img).numpy()[0]
    label=model.predict(img)[0]
    return p.labels[label.argmax()]

    # for i in range(len(label)):
    #     if label[i]==1:
    #         return  p.labels[i]