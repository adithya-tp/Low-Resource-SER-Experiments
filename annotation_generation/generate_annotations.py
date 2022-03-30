# Generate annotations for easy consumption of various audio datasets
import os
import re
import pandas as pd
import sys
sys.path.append(".")
sys.path.append("..")

import constants.PATHS as PATHS
import constants.EMOTIONS as EMOTS


def generate_aesdd_greek_annotations(DATASET_BASE_PATH, ANNOTATIONS_SAVE_PATH, EMOTIONS):
    """
    Function to generate annotations for the AESDD audio dataset 
    comprising of audio samples from the low resource Greek language. 
    Specifically, we keep track of the wav_file_path and parent_folder 
    (which also plays the role of the label, that is, emotion)

    :param DATASET_BASE_PATH: Base path to the AESDD dataset directory
    :param ANNOTATIONS_SAVE_PATH: Output path to save generated annotations
    :param EMOTIONS: The different emotions that are also the names of subfolders in dataset
    """
    df_list = [] # Dataframe as a list
    for emotion in EMOTIONS:
        for file in os.listdir(os.path.join(DATASET_BASE_PATH, emotion)):
            df_list.append([file, emotion])
    
    df = pd.DataFrame(df_list, columns=["file_path", "folder_or_label"])
    df.to_csv(ANNOTATIONS_SAVE_PATH, index=False)

def generate_emovo_italian_annotations(DATASET_BASE_PATH, ANNOTATIONS_SAVE_PATH, EMOTIONS):
    """
    Function to generate annotations for the EMOVO audio dataset 
    comprising of audio samples from the low resource Italian language. 
    Specifically, we keep track of the wav_file_path and parent_folder 
    (which also plays the role of the label, that is, emotion)

    :param DATASET_BASE_PATH: Base path to the EMOVO dataset directory
    :param ANNOTATIONS_SAVE_PATH: Output path to save generated annotations
    :param EMOTIONS: Dictionary that maps Italian emotion abbreviations to english labels
    """

    FOLDER_REGEX = "[mf][123]"

    df_list = []
    for folder in os.listdir(DATASET_BASE_PATH):
        if re.search(FOLDER_REGEX, folder):
            FOLDER_PATH = os.path.join(DATASET_BASE_PATH, folder)
            for file in os.listdir(FOLDER_PATH):
                emotion_abbreviation = file.split("-")[0]
                df_list.append([file, folder, EMOTIONS[emotion_abbreviation]])
    df = pd.DataFrame(df_list, columns=["file_path", "folder", "emotion"])
    df.to_csv(ANNOTATIONS_SAVE_PATH, index=False)


def main():
    output_path = 'annotations/annotations.csv'
    
    # Create annotations folder in dataset if does not exist
    os.makedirs(os.path.join(PATHS.AESDD_DIR, os.path.dirname(output_path)), exist_ok=True)
    os.makedirs(os.path.join(PATHS.EMOVO_DIR, os.path.dirname(output_path)), exist_ok=True)
    
    generate_aesdd_greek_annotations(
        DATASET_BASE_PATH=PATHS.AESDD_DIR,
        ANNOTATIONS_SAVE_PATH=os.path.join(PATHS.AESDD_DIR, output_path),
        EMOTIONS=EMOTS.AESDD_EMOTIONS
    )

    generate_emovo_italian_annotations(
        DATASET_BASE_PATH=PATHS.EMOVO_DIR,
        ANNOTATIONS_SAVE_PATH=os.path.join(PATHS.EMOVO_DIR, output_path),
        EMOTIONS=EMOTS.EMOVO_EMOTIONS
    )


if __name__ == '__main__':
    main()