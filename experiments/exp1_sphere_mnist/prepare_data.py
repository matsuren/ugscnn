'''Module to generate the spherical mnist data set'''

import sys; sys.path.append("../../meshcnn")
import gzip
import pickle
import numpy as np
import argparse
from torchvision import datasets
from projection_helper import img2ERP, erp2sphere
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--bandwidth",
                        help="the bandwidth of the S2 signal",
                        type=int,
                        default=30,
                        required=False)
    parser.add_argument("--noise",
                        help="the rotational noise applied on the sphere",
                        type=float,
                        default=1.0,
                        required=False)
    parser.add_argument("--chunk_size",
                        help="size of image chunk with same rotation",
                        type=int,
                        default=500,
                        required=False)
    parser.add_argument("--mnist_data_folder",
                        help="folder for saving the mnist data",
                        type=str,
                        default="MNIST_data",
                        required=False)
    parser.add_argument("--output_file",
                        help="file for saving the data output (.gz file)",
                        type=str,
                        default="mnist_ico3.gzip",
                        required=False)
    parser.add_argument("--no_rotate_train",
                        help="do not rotate train set",
                        dest='no_rotate_train', action='store_true')
    parser.add_argument("--no_rotate_test",
                        help="do not rotate test set",
                        dest='no_rotate_test', action='store_true')
    parser.add_argument("--mesh_file",
                        help="path to mesh file",
                        type=str,
                        default="mesh_files/icosphere_3.pkl")
    parser.add_argument("--direction", 
                        help="projection direction [NP/EQ] : North Pole / Equator",
                        type=str,
                        choices=["NP", "EQ"],
                        default="EQ")
    
    args = parser.parse_args()

    print("getting mnist data")
    trainset = datasets.MNIST(root=args.mnist_data_folder, train=True, download=True)
    testset = datasets.MNIST(root=args.mnist_data_folder, train=False, download=True)
    mnist_train = {}
    mnist_train['images'] = trainset.data.numpy()
    mnist_train['labels'] = trainset.targets.numpy()
    mnist_test = {}
    mnist_test['images'] = testset.data.numpy()
    mnist_test['labels'] = testset.targets.numpy()

    # result
    dataset = {}

    no_rotate = {"train": args.no_rotate_train, "test": args.no_rotate_test}
    outshape = (2 * args.bandwidth, 2 * args.bandwidth)

    for label, data in zip(["train", "test"], [mnist_train, mnist_test]):
        print("projecting {0} data set".format(label))
        projections = []
        for img in tqdm(data['images']):
            if not no_rotate[label]:
                h_rot = np.random.uniform(-180, 180)
                v_rot = np.random.randint(-90, 90)
            else:
                h_rot = 0
                v_rot = 0
            x = img2ERP(img, v_rot=v_rot, h_rot=h_rot, outshape=outshape)
            projections.append(x)
        dataset[label] = {
            'images': np.array(projections),
            'labels': data['labels']
        }

    x_train = dataset['train']['images']
    x_test = dataset['test']['images']
    y_train = dataset['train']['labels']
    y_test = dataset['test']['labels']
    p = pickle.load(open(args.mesh_file, "rb"))
    V = p['V']
    F = p['F']

    # whether to project to NP (north pole) or EQ (equator)
    cos45 = np.cos(np.pi / 2)
    sin45 = np.sin(np.pi / 2)
    if args.direction == "EQ":
        x_rot_mat = np.array([[1, 0, 0], [0, cos45, -sin45], [0, sin45, cos45]])
        V = V.dot(x_rot_mat)

    x_train_s2 = []
    print("Converting training set...")

    for i in tqdm(range(x_train.shape[0])):
        x_train_s2.append(erp2sphere(x_train[i], V))

    x_test_s2 = []
    print("Converting test set...")
    for i in tqdm(range(x_test.shape[0])):
        x_test_s2.append(erp2sphere(x_test[i], V))

    x_train_s2 = np.stack(x_train_s2, axis=0)
    x_test_s2 = np.stack(x_test_s2, axis=0)

    d = {"train_inputs": x_train_s2,
         "train_labels": y_train,
         "test_inputs": x_test_s2,
         "test_labels": y_test}

    pickle.dump(d, gzip.open(args.output_file, "wb"))


if __name__ == '__main__':
    main()
