from mxtdot_dataset_PIL import MxtDotDataset
import utils
import config
import config_server as config
import torch
from torch.utils.data import DataLoader
from utils import collate_fn
import models
from engine import train_one_epoch, evaluate
# import torchvision.transforms as T
import transforms as T
import os

def visualize(dataset, num_viz):
    for i in range(num_viz):
        image, target = dataset[i]
        utils.visualize_sample(image, target, config.CLASSES)

def get_transform(train):
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    # if train:
    #     transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def get_datasets_and_dataloaders():
    dataset_train = MxtDotDataset(config.TRAIN_DIR_IMGS, config.TRAIN_DIR_XMLS, config.CLASSES, get_transform(train=True))
    dataset_test = MxtDotDataset(config.TEST_DIR_IMGS, config.TEST_DIR_XMLS, config.CLASSES, get_transform(train=False))

    data_loader_train = DataLoader(
        dataset_train,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )
    data_loader_test = DataLoader(
        dataset_test,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    print(f"Number of training samples: {len(dataset_train)}")
    print(f"Number of validation samples: {len(dataset_test)}\n")

    return dataset_train, dataset_test, data_loader_train, data_loader_test

def train(model,device, data_loader_train, data_loader_test, steps_counter):
    model.to(device)
    # data_loader_train.to(device)
    # data_loader_test.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=config.SCHEDULER_STEP_SIZE,gamma=config.SCHEDULER_GAMMA)

    steps_per_epoch = len(data_loader_train.dataset) // config.BATCH_SIZE
    previous_steps_count = steps_counter
    
    
    print(f"\nSteps per epoch: {len(data_loader_train.dataset) // config.BATCH_SIZE}")
    print(f"Save frequency: Every {config.TRAIN_STEPS_SAVE_FREQ} minibatches")
    print(f"Step counter: {previous_steps_count}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Base learning rate: {config.LEARNING_RATE}")
    print(f"Learning rate decay: {config.SCHEDULER_GAMMA*100}% every {config.SCHEDULER_STEP_SIZE} epochs")
    print(f"Output dir: {config.MODEL_OUTPUT_DIR}")

    print("\nStarting training...\n")

    for epoch in range(config.NUM_STEPS_TRAINING):

        train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=config.TRAIN_PRINT_FREQ)
        lr_scheduler.step()

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch
        }

        print(f"steps: {max(epoch * steps_per_epoch, steps_per_epoch) + steps_counter}")
        print(f"diff: {(max(epoch * steps_per_epoch, steps_per_epoch) + steps_counter) - previous_steps_count}")

        if((max(epoch * steps_per_epoch, steps_per_epoch) + steps_counter) - previous_steps_count > config.TRAIN_STEPS_SAVE_FREQ):
            previous_steps_count = max(epoch * steps_per_epoch, steps_per_epoch) + steps_counter

            print(f"\nSaving model at training step {previous_steps_count}\n")

            torch.save(model.state_dict(), os.path.join(config.MODEL_OUTPUT_DIR, f"model_{previous_steps_count}_steps.pth"))
            
            # utils.save_on_master(checkpoint, os.path.join(config.MODEL_OUTPUT_DIR, f"model_{previous_steps_count}_steps.pth"))
            # utils.save_on_master(checkpoint, os.path.join(config.MODEL_OUTPUT_DIR, "checkpoint.pth"))

            evaluate(model, data_loader_test, device=device)

        if epoch*steps_per_epoch>=config.NUM_STEPS_TRAINING:
            previous_steps_count = max(epoch * steps_per_epoch, steps_per_epoch) + steps_counter
            print(f"Training finished after {epoch+1} epochs with {previous_steps_count} steps")
            torch.save(model.state_dict(), os.path.join(config.MODEL_OUTPUT_DIR, f"model_{previous_steps_count}_steps.pth"))
            torch.save(model.state_dict(), os.path.join(config.MODEL_OUTPUT_DIR, f"checkpoint.pth"))
            break

        torch.save(model.state_dict(), os.path.join(config.MODEL_OUTPUT_DIR, f"checkpoint_{epoch}.pth"))
        print(f"\nSaving model after epoch {epoch}\n")

        evaluate(model, data_loader_test, device=device)

    return model


def main():

    dataset_train, dataset_test, data_loader_train, data_loader_test = get_datasets_and_dataloaders()

    # https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

    print(f"CUDA available: {torch.cuda.is_available()}")
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cpu')

    model = models.create_faster_rcnn_resnet50_fpn(config.NUM_CLASSES)
    
    steps_counter = 0

    if config.LOAD_CHECKPOINT_PATH != "NONE":
        model.load_state_dict(torch.load(config.LOAD_CHECKPOINT_PATH))
        steps_counter = int(config.LOAD_CHECKPOINT_PATH.split("_")[-2])
        print(f"\nLoading checkpoint with {steps_counter} steps")

    # model = torch.nn.DataParallel(model)

    trained_model = train(model,device,data_loader_train,data_loader_test,steps_counter)
    

if __name__=="__main__":
    main()