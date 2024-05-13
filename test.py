import argparse
import json
import pathlib

import timm
import torch
import torchvision.transforms as transforms
import torchvision
import tqdm
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import Subset

from evaluation import compute_embedding, compute_knn, compute_logReg
from utils import Head, DINOLoss, MultiCropWrapper, clip_gradients
from DataAugmentation import DataAugmentation
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
import os



def main():
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


    parser = argparse.ArgumentParser(
        "DINO testing CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-b", "--batch-size", type=int, default=32)
    parser.add_argument(
        "-d", "--device", type=str, choices=("cpu", "cuda"), default="cuda"
    )
    parser.add_argument("-l", "--logging-freq", type=int, default=200)
    #parser.add_argument("-c", "--n-crops", type=int, default=4)
    #parser.add_argument("-e", "--n-epochs", type=int, default=100)
    parser.add_argument("-o", "--out-dim", type=int, default=1024)
    parser.add_argument("-t", "--tensorboard-dir", type=str, default="logs")
    #parser.add_argument("--clip-grad", type=float, default=2.0)
    parser.add_argument("--norm-last-layer", action="store_true")
    parser.add_argument("--batch-size-eval", type=int, default=32)
    parser.add_argument("--subset", type=int, default=0)
    parser.add_argument("--student-temp", type=float, default=0.1)
    parser.add_argument("--pretrained", action="store_true")
    # parser.add_argument("-m", "--best-model-file", type=str, default="latest_model.pth")

    #parser.add_argument("-w", "--weight-decay", type=float, default=0.4)
    args = parser.parse_args()


    #device = torch.device(args.device)

    device = torch.device('cpu')
    num_workers = 4


    vit_model_name, embedding_dim = "vit_deit_small_patch16_224", 384
    student_vit = timm.create_model(vit_model_name, pretrained=args.pretrained)

    student_model = MultiCropWrapper(
        student_vit,
        Head(embedding_dim, args.out_dim, norm_last_layer=args.norm_last_layer),
    ).to(device)
   


    # Preload
    # print(f'Preloading model {args.best_model_file}')
    state = torch.load('best_model.pth',map_location=torch.device('cpu'))
    #print("state:",state)
    student_model.load_state_dict(state.state_dict())

    plain_transform = transforms.Compose([  # Transform PIL image to tensor
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        transforms.Resize((224, 224)),
    ])

    # Used for Visualizing embeddings after running training set result through KNN classifier
    train_dataset_plain = torchvision.datasets.CIFAR10(root='./data',
                                                       download=True,
                                                       train=True,
                                                       transform=plain_transform)
    test_dataset_plain = torchvision.datasets.CIFAR10(root='./data',
                                                     download=True,
                                                     train=False,
                                                     transform=plain_transform)
    

    train_loader_plain = DataLoader(
        train_dataset_plain,
        batch_size=args.batch_size_eval,
        drop_last=False,
        num_workers=num_workers,
    )
    test_loader_plain = DataLoader(  # loader of test data to generate embeddings
        test_dataset_plain,
        batch_size=args.batch_size_eval,
        drop_last=False,
        num_workers=num_workers,
    )

    #num_steps = state['num_steps']

    logging_dir = pathlib.Path(args.tensorboard_dir)
    writer = SummaryWriter(logging_dir)
    writer.add_text("arguments", json.dumps(vars(args)))



    current_accuracy = compute_knn(
                    student_model.backbone, train_loader_plain, test_loader_plain)
    writer.add_scalar(
                    "knn-accuracy", current_accuracy['accuracy'])
    writer.add_scalar(
                    "knn-fscore", current_accuracy['weighted avg']['f1-score'])
    print("Test Accuracy:", current_accuracy['accuracy'])

    file = open("Test_results.txt", "a")
    file.write(json.dumps(current_accuracy)+"\n")
    file.close()

if __name__ == "__main__":
    main()


