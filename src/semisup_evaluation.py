import argparse
from collections import defaultdict, OrderedDict

import numpy as np
import torch
import torch.nn as nn

from easyfl.datasets.data import CIFAR100
from eval_dataset import get_data_loaders, get_semi_supervised_data_loaders
from model import get_encoder_network


def inference(loader, model, device):
    feature_vector = []
    labels_vector = []
    model.eval()
    for step, (x, y) in enumerate(loader):
        x = x.to(device)

        # get encoding
        with torch.no_grad():
            h = model(x)

        h = h.squeeze()
        h = h.detach()

        feature_vector.extend(h.cpu().detach().numpy())
        labels_vector.extend(y.numpy())

        if step % 5 == 0:
            print(f"Step [{step}/{len(loader)}]\t Computing features...")

    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    print("Features shape {}".format(feature_vector.shape))
    return feature_vector, labels_vector


def get_features(model, train_loader, test_loader, device):
    train_X, train_y = inference(train_loader, model, device)
    test_X, test_y = inference(test_loader, model, device)
    return train_X, train_y, test_X, test_y


def create_data_loaders_from_arrays(X_train, y_train, X_test, y_test, batch_size):
    train = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train)
    )
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=False
    )

    test = torch.utils.data.TensorDataset(
        torch.from_numpy(X_test), torch.from_numpy(y_test)
    )
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader


def test_result(test_loader, logreg, device, model_path, criterion):
    # Test fine-tuned model
    print("### Calculating final testing performance ###")
    logreg.eval()
    metrics = defaultdict(list)
    for step, (h, y) in enumerate(test_loader):
        h = h.to(device)
        y = y.to(device)

        outputs = logreg(h)
        loss = criterion(outputs, y)

        # calculate accuracy and save metrics
        accuracy = (outputs.argmax(1) == y).sum().item() / y.size(0)
        metrics["Loss/test"].append(loss.item())
        metrics["Accuracy/test"].append(accuracy)

    print(f"Final test performance: " + model_path)
    for k, v in metrics.items():
        print(f"{k}: {np.array(v).mean():.4f}")
    return np.array(metrics["Accuracy/test"]).mean(), np.array(metrics["Loss/test"]).mean()


if __name__ == "__main__":

    print("#########################################")
    print("### Semi-Supervised Evaluation Script ###")
    print("#########################################")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar10", type=str)
    parser.add_argument("--model_path", type=str, help="Path to pre-trained model (e.g. model-10.pt)")
    parser.add_argument('--model', default='byol', type=str, help='name of the network')
    parser.add_argument("--image_size", default=32, type=int, help="Image size")
    parser.add_argument("--learning_rate", default=3e-4, type=float, help="Initial learning rate.")
    parser.add_argument("--batch_size", default=512, type=int, help="Batch size for training.")
    parser.add_argument("--num_epochs", default=200, type=int, help="Number of epochs to train for.")
    parser.add_argument("--encoder_network", default="resnet34", type=str, help="Encoder network architecture.")
    parser.add_argument("--num_workers", default=8, type=int, help="Number of data workers (caution with nodes!)")
    parser.add_argument('--gpu', default=2, type=int)
    parser.add_argument("--fc", default="identity", help="options: identity, remove")
    parser.add_argument("--label_ratio", default=0.01, type=float, help="options: 0.01, 0.1")
    parser.add_argument("--data_dist", default="class", help="options: iid, class, dir")
    args = parser.parse_args()
    print(args)

    device = torch.device("cuda",args.gpu) if torch.cuda.is_available() else torch.device("cpu")

    # get data loaders
    train_loader, test_loader = get_semi_supervised_data_loaders(args.dataset, args.data_dist, 5, args.label_ratio, args.batch_size)
    # get model
    resnet = get_encoder_network(args.model, args.encoder_network)
    load_model = torch.load(args.model_path, map_location=device)
    new_model = OrderedDict()
    for k, v in load_model.items():
        if k[:15] == 'online_encoder.':
            name = k[15:]
            new_model[name] = v
    resnet.load_state_dict(new_model)
    # resnet.load_state_dict(load_model)
    resnet = resnet.to(device)
    if args.encoder_network == 'cnn':
        num_features = list(resnet.children())[-2].out_features
    else:
        num_features = list(resnet.children())[-1].in_features
    if args.fc == "remove":
        resnet = nn.Sequential(*list(resnet.children())[:-1])  # throw away fc layer
    else:
        resnet.fc = nn.Identity()

    n_classes = 10
    if args.dataset == "cifar100": #CIFAR100:
        n_classes = 100

    # fine-tune model
    logreg = nn.Sequential(resnet, nn.Linear(num_features, n_classes))
    logreg = logreg.to(device)

    for p in logreg.parameters():
        p.requires_grad = True

    # loss / optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=logreg.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    # Train fine-tuned model
    logreg.train()

    for epoch in range(args.num_epochs):
        metrics = defaultdict(list)
        for step, (h, y) in enumerate(train_loader):
            h = h.to(device)
            y = y.to(device)

            outputs = logreg(h)

            loss = criterion(outputs, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # calculate accuracy and save metrics
            accuracy = (outputs.argmax(1) == y).sum().item() / y.size(0)
            metrics["Loss/train"].append(loss.item())
            metrics["Accuracy/train"].append(accuracy)

        # scheduler.step()

        print(f"Epoch [{epoch}/{args.num_epochs}]: " + "\t".join(
            [f"{k}: {np.array(v).mean()}" for k, v in metrics.items()]))

        if epoch % 10 == 0:
            print("======epoch {}======".format(epoch))
            test_result(test_loader, logreg, device, args.model_path, criterion)
    result1, _ = test_result(test_loader, logreg, device, args.model_path, criterion)
    result1 = result1 * 100
    print(f"{result1:.2f}")
