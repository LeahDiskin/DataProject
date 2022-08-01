
import string

import pandas as pd

from Utils import params as p
import numpy as np
from tensorflow import keras
import numpy as np
print(f"Numpy: {np.version.version}")



model=keras.models.load_model(r"C:\Users\IMOE001\Downloads\model")

success_rate = pd.read_csv(p.results_csv_path)

# list1=list(success_rate['Unnamed: 0'].array)
# list2=list(success_rate['Unnamed: 1'].array)
# list3=list(success_rate['success rates'].array)
# hier_index=list(zip(list1,list2))
# dfdf=pd.DataFrame(list3,index=hier_index)


def predict(img:np.ndarray,i)->string:
    img = img.astype('float32')
    img /= 255
    img=img.reshape(-1,img.shape[0],img.shape[1],img.shape[2])
    # label=model(img).numpy()[0]

    label_score=model.predict(img)[0]

    for j in range(0,i):
        label_score[label_score.argmax()]=0

    ind_label=label_score.argmax()
    if i==0:
        max_score = np.amax(label_score)
        if max_score==1.0:
            pred_success_rate=100
        else:
            pred_success_rate=success_rate[success_rate['Unnamed: 0']==ind_label].loc[success_rate['Unnamed: 1']==int(max_score*10)]['success rates'].iloc[0]
        ind = int(pred_success_rate / 10)
        sentence = p.dict[ind]
        pred_success_rate=str(pred_success_rate)+'%'

    else:
        pred_success_rate=""
        sentence=""


    return p.labels[label_score.argmax()], (pred_success_rate),sentence