# Generate annotations for easy consumption of various audio datasets
import os
import pandas as pd
import sys
sys.path.append(".")
sys.path.append("..")

import constants.PATHS as PATHS
import constants.EMOTIONS as EMOTS


def generate_aesdd_greek_annotations(DATASET_BASE_PATH, ANNOTATIONS_SAVE_PATH, EMOTIONS):
    """
    Function to generate annotations for the low resource AESDD 
    audio dataset for the Greek Language. Specifically we keep track
    of the wav_file_path and parent_folder (which also plays the role of 
    the label, that is, emotion)

    :param DATASET_BASE_PATH: Base path to the AESDD dataset directory
    :param ANNOTATIONS_SAVE_PATH: Output path to save generated annotations
    :param EMOTIONS: The different emotions that are also the names of subfolders in dataset
    """
    paths_and_labels = []
    for emotion in EMOTIONS:
        for file in os.listdir(os.path.join(DATASET_BASE_PATH, emotion)):
            paths_and_labels.append([file, emotion])
    
    df = pd.DataFrame(paths_and_labels, columns=["file_path", "folder_or_label"])
    df.to_csv(ANNOTATIONS_SAVE_PATH)


def main():
    output_path = 'annotations/annotations.csv'
    
    # Create annotations folder in dataset if does not exist
    os.makedirs(os.path.join(PATHS.AESDD_DIR, os.path.dirname(output_path)), exist_ok=True)
    
    generate_aesdd_greek_annotations(
        DATASET_BASE_PATH=PATHS.AESDD_DIR,
        ANNOTATIONS_SAVE_PATH=os.path.join(PATHS.AESDD_DIR, output_path),
        EMOTIONS=EMOTS.AESDD_EMOTIONS
    )

if __name__ == '__main__':
    main()