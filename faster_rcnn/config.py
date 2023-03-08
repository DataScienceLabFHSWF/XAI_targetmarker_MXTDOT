# https://debuggercafe.com/custom-object-detection-using-pytorch-faster-rcnn/

import torch

BATCH_SIZE = 4

NUM_EPOCHS = 100 # number of epochs to train for
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# TRAIN_DIR_IMGS = "/home/dan/FH/Paper/private/images/v3/crops/train"
# TRAIN_DIR_XMLS = "/home/dan/FH/Paper/public/annotations/paper/crops/train_grep"

# TEST_DIR_IMGS = "/path/to/data/targetmarker/test/imgs"
# TEST_DIR_XMLS = "/path/to/data/targetmarker/test/annotations"
TEST_DIR_IMGS = "../data/imgs_annot/frcnn50/test/imgs"
TEST_DIR_XMLS = "../data/imgs_annot/frcnn50/test/annotations"
SAVEPOINT_LOC = "../data/checkpoints/frcnn50/1/checkpoint.pth"


MODEL_SAVE_DIR = "/path/to/save/dir" # only for training

# classes: 0 index is reserved for background
CLASSES = [
    'background', 'MXT90', 'DOT'
]
NUM_CLASSES = 3
# whether to visualize images after crearing the data loaders
VISUALIZE_TRANSFORMED_IMAGES = False
# location to save model and plots
OUT_DIR = '../outputs'
SAVE_PLOTS_EPOCH = 2 # save loss plots after these many epochs
SAVE_MODEL_EPOCH = 2 # save model after these many epochs
LEARNING_RATE = 0.005
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
SCHEDULER_STEP_SIZE = 3
SCHEDULER_GAMMA = 0.1
TRAIN_PRINT_FREQ = 10 # print progress every x batches
TRAIN_STEPS_SAVE_FREQ = 1500
NUM_STEPS_TRAINING = 20000
ANCHORSIZES = [8, 16, 32, 64, 128]
SCALESIZES = [0.5, 1.0, 2.0]
