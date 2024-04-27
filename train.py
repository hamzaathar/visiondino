import argparse
import json
import pathlib

import timm
import torch
import torchvision.transforms as transforms
import tqdm
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder

from evaluation import compute_embedding, compute_knn
from utils import DataAugmentation, Head, Loss, MultiCropWrapper, clip_gradients


def main():
    parser = argparse.ArgumentParser(
        "DINO training CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-b", "--batch-size", type=int, default=32)
    parser.add_argument(
        "-d", "--device", type=str, choices=("cpu", "cuda"), default="cpu"
    )
    parser.add_argument("-l", "--logging-freq", type=int, default=200)
    parser.add_argument("--momentum-teacher", type=int, default=0.9995)
    parser.add_argument("-c", "--n-crops", type=int, default=4)
    parser.add_argument("-e", "--n-epochs", type=int, default=100)
    parser.add_argument("-o", "--out-dim", type=int, default=1024)
    parser.add_argument("-t", "--tensorboard-dir", type=str, default="logs")
    parser.add_argument("--clip-grad", type=float, default=2.0)
    parser.add_argument("--norm-last-layer", action="store_true")
    parser.add_argument("--batch-size-eval", type=int, default=64)
    parser.add_argument("--teacher-temp", type=float, default=0.04)
    parser.add_argument("--student-temp", type=float, default=0.1)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("-w", "--weight-decay", type=float, default=0.4)

    args = parser.parse_args()
    print(vars(args))

    # Dataset paths
    train_dataset_path = pathlib.Path("data/imagenette2-320/train")
    val_dataset_path = pathlib.Path("data/imagenette2-320/val")
    labels_path = pathlib.Path("data/imagenette_labels.json")

    # Tensorboard logging
    logging_dir = pathlib.Path(args.tensorboard_dir)
    writer = SummaryWriter(logging_dir)
    writer.add_text("arguments", json.dumps(vars(args)))

    # Device setup
    device = torch.device(args.device)
    num_workers = 4

    # Load label mapping
    with labels_path.open("r") as f:
        label_mapping = json.load(f)

    # Data transformations
    augmentation_transform = DataAugmentation(size=224, n_local_crops=args.n_crops - 2)#Size of input image with numb of local crops -2 due to the 2 global crops
    plain_transform = transforms.Compose([# Transform PIL image to tensor
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        transforms.Resize((224, 224)),
    ])

    # Define datasets and data loaders
    train_dataset_aug = ImageFolder(train_dataset_path, transform=augmentation_transform) # Used for training, augmented dataset
    train_dataset_plain = ImageFolder(train_dataset_path, transform=plain_transform) # Used for Visualizing embeddings after running training set result through KNN classifier
    val_dataset_plain = ImageFolder(val_dataset_path, transform=plain_transform)

    if train_dataset_plain.classes != val_dataset_plain.classes:
        raise ValueError("Inconsistent classes")

    train_loader_aug = DataLoader(
        train_dataset_aug,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    train_loader_plain = DataLoader(
        train_dataset_plain,
        batch_size=args.batch_size_eval,
        drop_last=False,
        num_workers=num_workers,
    )
    val_loader_plain = DataLoader( # loader of validate data to generate embeddings
        val_dataset_plain,
        batch_size=args.batch_size_eval,
        drop_last=False,
        num_workers=num_workers,
    )
    val_loader_plain_subset = DataLoader( # subset of validate to generate embeddings, used for visualization on TensorBoard
        val_dataset_plain,
        batch_size=args.batch_size_eval,
        drop_last=False,
        sampler=SubsetRandomSampler(list(range(0, len(val_dataset_plain), 50))),
        num_workers=num_workers,
    )


    # Model setup
    vit_model_name, embedding_dim = "vit_deit_small_patch16_224", 384
    student_model = MultiCropWrapper(
        timm.create_model(vit_model_name, pretrained=args.pretrained),
        Head(embedding_dim, args.out_dim, norm_last_layer=args.norm_last_layer),
    ).to(device)
    teacher_model = MultiCropWrapper(
        timm.create_model(vit_model_name, pretrained=args.pretrained),
        Head(embedding_dim, args.out_dim),
    ).to(device)

    teacher_model.load_state_dict(student_model.state_dict())# make sure teacher and student model weights are identical at the start ( Wont hold after training as teacher weights will be a weighted avereage of the student weights)

    for param in teacher_model.parameters(): #Make teachers params untrainable (save memory), as we only train student model.
        param.requires_grad = False

    # Loss setup
    loss_instance = Loss(args.out_dim, teacher_temp=args.teacher_temp, student_temp=args.student_temp).to(device)
    learning_rate = 0.0005 * args.batch_size / 256
    optimizer = torch.optim.AdamW(
        student_model.parameters(),
        lr=learning_rate,
        weight_decay=args.weight_decay,
    )

    # Training loop
    num_batches = len(train_dataset_aug) // args.batch_size
    best_accuracy = 0
    num_steps = 0

    for epoch in range(args.n_epochs):
        for batch_idx, (images, _) in tqdm.tqdm(enumerate(train_loader_aug), total=num_batches): # as this is self supervised learning, we dont use the labels from the loader
            if num_steps % args.logging_freq == 0:
                student_model.eval()

                # Embeddings
                embs, imgs, labels_ = compute_embedding(student_model.backbone, val_loader_plain_subset)
                writer.add_embedding(
                    embs,
                    metadata=[label_mapping[label] for label in labels_],
                    label_img=imgs,
                    global_step=num_steps,
                    tag="embeddings",
                )

                # KNN evaluation
                current_accuracy = compute_knn(student_model.backbone, train_loader_plain, val_loader_plain)  # compute accuracy of model using KNN
                writer.add_scalar("knn-accuracy", current_accuracy, num_steps)

                if current_accuracy > best_accuracy: #Save best model
                    torch.save(student_model, logging_dir / "best_model.pth")
                    best_accuracy = current_accuracy

                student_model.train()

            images = [img.to(device) for img in images]

            teacher_output = teacher_model(images[:2]) #Teacher model only gets the 2 global cropped images
            student_output = student_model(images) #Student model gets all the images

            loss = loss_instance(student_output, teacher_output) # Compute loss between the Student and teacher model outputs

            optimizer.zero_grad()#Backprop
            loss.backward()
            clip_gradients(student_model, args.clip_grad)
            optimizer.step() #Student model weights updated

            #Now update the Teacher model based on the Student Model
            with torch.no_grad():
                for student_param, teacher_param in zip(student_model.parameters(), teacher_model.parameters()): #Update parameter by parrameter
                    #Update using the exponentional moving average
                    teacher_param.data.mul_(args.momentum_teacher)
                    teacher_param.data.add_((1 - args.momentum_teacher) * student_param.detach().data)

            writer.add_scalar("train_loss", loss, num_steps) #Add to tensorboard

            num_steps += 1


if __name__ == "__main__":
    main()
