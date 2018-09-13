import sys
sys.path.append("/home/claudio/Documents/GitHub/datascienceworkshop-pneumonia")

from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import pandas as pd
from config import TRAIN_IMAGES
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


def _get_feature_values(img_path, model):
    """ Get the feature values for a specific image out of pre-trained model """
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return model.predict(x)[0].reshape(-1)


def _create_pretrained_feature_df(class_directory):
    """ Create features dataframe based on pretrained model """
    features = []
    for n, file in enumerate(os.listdir(class_directory)):
        print(f"Scoring image {n}")
        full_path = os.path.join(class_directory, file)
        features.append(_get_feature_values(full_path, model))
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
    model = VGG16(weights='imagenet', include_top=False)

    pos_df = _create_pretrained_feature_df(os.path.join(TRAIN_IMAGES, "positive"))
    pos_df["Target"] = 1

    neg_df = _create_pretrained_feature_df(os.path.join(TRAIN_IMAGES, "negative"))
    neg_df["Target"] = 0

    full_df = pd.concat([pos_df, neg_df])

    rf_trained = _build_basic_rf(full_df)