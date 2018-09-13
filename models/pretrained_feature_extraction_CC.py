import sys
sys.path.append("/home/claudio/Documents/GitHub/datascienceworkshop-pneumonia")

from keras.applications.vgg16 import                VGG16
from keras.applications.vgg19 import                VGG19
from keras.applications.resnet50 import             ResNet50
from keras.applications.xception import             Xception
from keras.applications.inception_resnet_v2 import  InceptionResNetV2
from keras.applications.inception_v3 import         InceptionV3
from keras.applications.mobilenet import            MobileNet
from keras.applications.mobilenetv2 import          MobileNetV2
from keras.applications.densenet import             DenseNet121
from keras.applications.densenet import             DenseNet169
from keras.applications.densenet import             DenseNet201
from keras.applications.nasnet import               NASNetLarge
from keras.applications.nasnet import               NASNetMobile
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import pandas as pd
from config import TRAIN_IMAGES
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

class Model(object):

    def __init__(self, name, architecture, input_size, include_top, pooling):
        self.Name = name
        self.Architecture = architecture
        self.Input_size = input_size
        self.Include_top = include_top
        self.Pooling = pooling
        self.Model = None

    def CreateModel(self):
        self.Model = self.Architecture(weights='imagenet', include_top=self.Include_top, pooling=self.Pooling)


architectures = {
"VGG16":VGG16,
"VGG19":VGG19,
"ResNet50":ResNet50,
"Xception":Xception,
"InceptionResNetV2":InceptionResNetV2,
"InceptionV3":InceptionV3,
"MobileNet":MobileNet,
"MobileNetV2":MobileNetV2,
"DenseNet121":DenseNet121,
"DenseNet169":DenseNet169,
"DenseNet201":DenseNet201,
"NASNetLarge":NASNetLarge,
"NASNetMobile":NASNetMobile
}

input_size = {
"VGG16":(224,224),
"VGG19":(224,224),
"ResNet50":(224,224),
"Xception":(299,299),
"InceptionResNetV2":(299,299),
"InceptionV3":(299,299),
"MobileNet":(224,224),
"MobileNetV2":(224,224),
"DenseNet121":(224,224),
"DenseNet169":(224,224),
"DenseNet201":(224,224),
"NASNetLarge":(331,331),
"NASNetMobile":(224,224)
}

def _collect_models():
    models = {}
    for key,value in architectures.items() :
        models[key] = Model(key,value,input_size[key],False,'max')
    return models

def _generate_model(models):
    for key,value in models.items():
        #pos features
        pos_df = _create_pretrained_feature_df(os.path.join(TRAIN_IMAGES, "positive"),value, partition=0.005,target_size=input_size[key])
        pos_df["Target"] = 1
        #neg features
        neg_df = _create_pretrained_feature_df(os.path.join(TRAIN_IMAGES, "negative"),value, partition=0.005,target_size=input_size[key])
        neg_df["Target"] = 0
        #merge
        # full_df = pd.concat([pos_df, neg_df])
        #full_df.to_csv(path_or_buf=r'/home/claudio/Documents/output/{}'.format(key))

        del value.Model



def _get_feature_values(img_path, model,target_size):
    """ Get the feature values for a specific image out of pre-trained model """
    img = image.load_img(img_path, target_size=target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return model.predict(x)[0].reshape(-1)


def _create_pretrained_feature_df(class_directory, model, partition=1.0,target_size=(224, 224)):
    """ Create features dataframe based on pretrained model """
    features = []
    
    #instantiate model
    model.CreateModel()
    m = model.Model

    def enumImgPath(class_directory,partition):
        max = len(os.listdir(class_directory))
        i=0
        for n, file in enumerate(os.listdir(class_directory)):
            if i > int(max*partition):
                break
            i+=1
            yield file
    #
    #
    divisor = 50
    n_max = len(os.listdir(class_directory))
    """
    #for n, file in enumerate(enumImgPath(class_directory,partition)):
    #for n, file in enumerate(os.listdir(class_directory)):
        print(f"Scoring image {n}")
        full_path = os.path.join(class_directory, file)
        features.append(_get_feature_values(full_path, model,target_size=target_size))
    """
    for n, file in enumerate(enumImgPath(class_directory,partition)):
        print(f"Scoring image {n}")
        full_path = os.path.join(class_directory, file)
        features.append(_get_feature_values(full_path, m,target_size=target_size))
        if( (n+1) % (n_max/divisor) ==0 ):
            print('Scoring image {0}'.format(n/n_max))
            df = pd.DataFrame(np.stack(features))
            df.to_csv(path_or_buf=r'/home/claudio/Documents/output/{0}_{1:05d}.csv'.format(model.Name,n))
            features = []

    df = pd.DataFrame(np.stack(features))
    df.to_csv(path_or_buf=r'/home/claudio/Documents/output/{0}_{1:05d}.csv'.format(model.Name,n))
            
    return pd.DataFrame(np.stack(features))


def _build_basic_rf(df):
    """ Build basic random forest based on extracted pre-trained model """
    x_train, x_test, y_train, y_test = train_test_split(df[df.columns.difference(["Target"])], df["Target"])
    rf = RandomForestClassifier(n_estimators=100, max_depth=6, min_samples_leaf=15, n_jobs=-1, class_weight="balanced")
    rf.fit(x_train, y_train)
    train_preds = rf.predict_proba(x_train)
    test_preds = rf.predict_proba(x_test)
    print(f"AUC for train set is {roc_auc_score(y_train, train_preds[:,1])}")
    print(f"AUC for test set is {roc_auc_score(y_test, test_preds[:,1])}")
    return rf


if __name__ == "__main__":
    models = _collect_models()
    _generate_model(models)

"""
if __name__ == "__main__":
    model = VGG16(weights='imagenet', include_top=False)

    pos_df = _create_pretrained_feature_df(os.path.join(TRAIN_IMAGES, "positive"))
    pos_df["Target"] = 1

    neg_df = _create_pretrained_feature_df(os.path.join(TRAIN_IMAGES, "negative"))
    neg_df["Target"] = 0

    full_df = pd.concat([pos_df, neg_df])

    rf_trained = _build_basic_rf(full_df)
    """