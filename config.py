import os
import logging

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
TRAIN_DCM = os.path.join(DATA_DIR, "stage_1_train_images")
TEST_DCM = os.path.join(DATA_DIR, "stage_1_test_images")
TRAIN_IMAGES = os.path.join(DATA_DIR, "train_images")
TEST_IMAGES = os.path.join(DATA_DIR, "test_images")

LOGGER = logging.getLogger()
formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
