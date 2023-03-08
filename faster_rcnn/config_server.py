# https://debuggercafe.com/custom-object-detection-using-pytorch-faster-rcnn/

BATCH_SIZE = 16 # increase / decrease according to GPU memeory

NUM_EPOCHS = 50 # number of epochs to train for

# training images and XML files directory
FILES_DIR = "/media/datasciencefhswf/data2/fruits/images/"
# TRAIN_DIR_IMGS = "/media/dagie002/data3/targetmarker/paper/pytorch/images/v3/crops/train"
TRAIN_DIR_IMGS = "/media/datasciencefhswf/data2/targetmarker/paper/pytorch/images/v3/crops/train"
# TRAIN_DIR_IMGS = "/media/datasciencefhswf/data2/potholes/images/train"
# TRAIN_DIR_IMGS = "/media/datasciencefhswf/data2/fruits/images/train"
# TRAIN_DIR_XMLS = "/media/dagie002/data3/targetmarker/paper/pytorch/annotations/crops/train_grep"
TRAIN_DIR_XMLS = "/media/datasciencefhswf/data2/targetmarker/paper/pytorch/annotations/crops/train_grep"
# TRAIN_DIR_XMLS = "/media/datasciencefhswf/data2/potholes/annotations/train"
# TRAIN_DIR_XMLS = "/media/datasciencefhswf/data2/fruits/annotations/train"
# validation images and XML files directory
# TEST_DIR_IMGS = "/media/dagie002/data3/targetmarker/paper/pytorch/images/v3/crops/test"
TEST_DIR_IMGS = "/media/datasciencefhswf/data2/targetmarker/paper/pytorch/images/v3/crops/test"
# TEST_DIR_IMGS = "/media/datasciencefhswf/data2/potholes/images/test"
# TEST_DIR_IMGS = "/media/datasciencefhswf/data2/fruits/images/test"
# TEST_DIR_XMLS = "/media/dagie002/data3/targetmarker/paper/pytorch/annotations/crops/test_grep"
TEST_DIR_XMLS = "/media/datasciencefhswf/data2/targetmarker/paper/pytorch/annotations/crops/test_grep"
# TEST_DIR_XMLS = "/media/datasciencefhswf/data2/potholes/annotations/test"
# TEST_DIR_XMLS = "/media/datasciencefhswf/data2/fruits/annotations/test"
# MODEL_OUTPUT_DIR = "/media/dagie002/data3/targetmarker/paper/pytorch/models/faster_rcnn50/1"
MODEL_OUTPUT_DIR = "/media/datasciencefhswf/data2/targetmarker/paper/pytorch/models/faster_rcnn50/4"
# MODEL_OUTPUT_DIR = "/media/datasciencefhswf/data2/potholes/models/faster_rcnn50/1"
MODEL_OUTPUT_DIR = "/media/datasciencefhswf/data2/fruits/models/faster_rcnn50/1"

LOAD_CHECKPOINT_PATH = "NONE"

# LOAD_CHECKPOINT_PATH = "/media/datasciencefhswf/data2/targetmarker/paper/pytorch/models/faster_rcnn50/1/model_3040_steps.pth"

# classes: 0 index is reserved for background
CLASSES = [
     'background', 'MXT90', 'DOT'
]
NUM_CLASSES = 3

# CLASSES = [
#     'background', 'pothole'
#]
# NUM_CLASSES = 2

# CLASSES = [
#     'background', 'apple', 'orange', 'banana' 
#]
# NUM_CLASSES = 4

# whether to visualize images after crearing the data loaders
VISUALIZE_TRANSFORMED_IMAGES = False
# location to save model and plots
SAVE_PLOTS_EPOCH = 2 # save loss plots after these many epochs
SAVE_MODEL_EPOCH = 2 # save model after these many epochs
LEARNING_RATE = 0.001
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
SCHEDULER_STEP_SIZE = 10
SCHEDULER_GAMMA = 0.5
TRAIN_PRINT_FREQ = 10 # print progress every x batches
TRAIN_STEPS_SAVE_FREQ = 1500
NUM_STEPS_TRAINING = 20000

ANCHORSIZES = [8, 16, 32, 64, 128]
SCALESIZES = [0.5, 1.0, 2.0]
