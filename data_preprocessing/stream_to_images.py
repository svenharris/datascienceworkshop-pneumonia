import sys
sys.path.append("/home/claudio/Documents/GitHub/datascienceworkshop-pneumonia")


from config import TEST_DCM, TRAIN_DCM, TRAIN_IMAGES, TEST_IMAGES, DATA_DIR
import pydicom
from PIL import Image
import pandas as pd
import os
import shutil




def _get_metadata(dcm_file):
    """ Extracts image metadata from DCM file and converts into dictionary """
    raw_dict = {x.description(): x.value for x in dcm_file.iterall() if x.description() != "Pixel Data"}
    for key in raw_dict.copy().keys():
        new_key = key.replace("'", "").replace(" ", "_").lower()
        raw_dict[new_key] = raw_dict.pop(key)
    return raw_dict


def _convert_to_png(dcm_file):
    """ Converts DCM file type to PIL Image object """
    raw_pixels = dcm_file.pixel_array
    pil_image = Image.fromarray(raw_pixels)
    return pil_image


def _stream_to_dir(input_loc, output_loc):
    """ Converts all DCM files in a directory into PNGs with an associated metadata file """
    all_metas = []
    for file in os.listdir(input_loc):
        filepath = os.path.join(input_loc, file)
        dcm_obj = pydicom.read_file(filepath)
        png_image = _convert_to_png(dcm_obj)
        metadata = _get_metadata(dcm_obj)
        new_filename = file.split(".")[0] + ".png"
        png_image.save(os.path.join(output_loc, new_filename), "png")
        all_metas.append(metadata)

    meta_df = pd.DataFrame(all_metas)
    meta_df.to_csv(os.path.join(output_loc, "metadata.csv"))
    return meta_df


def _get_matching_files(file_names, dir):
    """ Get matching files that exist in the directory """
    return [os.path.join(dir, x) for x in os.listdir(dir) if x in file_names]


def _move_to_class_folder(png_dir, labels_loc):
    """ Looks at the metadata and sorts the images into subfolders based on their target class """
    metadata_df = pd.read_csv(labels_loc)
    metadata_df["png_names"] = metadata_df["patientId"].astype(str) + ".png"
    positive_obs = metadata_df.loc[metadata_df["Target"] == 1, "png_names"]
    negative_obs = metadata_df.loc[metadata_df["Target"] == 0, "png_names"]
    pos_paths = _get_matching_files(list(positive_obs), png_dir)
    neg_paths = _get_matching_files(list(negative_obs), png_dir)
    _move_files(pos_paths, os.path.join(png_dir, "positive"))
    _move_files(neg_paths, os.path.join(png_dir, "negative"))


def _move_files(file_paths, destination):
    """ Move list of files to a destination folder """
    for file in file_paths:
        shutil.move(file, destination)


if __name__ == "__main__":
    _stream_to_dir(TEST_DCM, TEST_IMAGES)
    _stream_to_dir(TRAIN_DCM, TRAIN_IMAGES)
    _move_to_class_folder(TRAIN_IMAGES, os.path.join(DATA_DIR, "stage_1_train_labels.csv"))
