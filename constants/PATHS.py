import os

GITHUB_REPO_NAME = 'Low-Resource-SER-Experiments'

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
while os.path.basename(os.path.abspath(ROOT_DIR)) != GITHUB_REPO_NAME:
    ROOT_DIR = os.path.join(ROOT_DIR, "..")

AESDD_DIR = os.path.join(ROOT_DIR, 'data/AESDD')
EMOVO_DIR = os.path.join(ROOT_DIR, 'data/EMOVO')