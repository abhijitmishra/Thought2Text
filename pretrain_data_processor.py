# This program converts BDD and Mind BIG Data to pt files to be used for training and validation
# Sources:
# 1. https://github.com/LKbrilliant/Brain-Download-Datasets/VisualQA
# 2. https://mindbigdata.com/opendb/MindBigData-IN-v1.06.zip
# 3. https://mindbigdata.com/opendb/MindBigData-Imagenet-IN.zip

# Input data: Formats
## BDD
# Multiple files, each for one session in this format
# COUNTER, AF3, T7, Pz, T8, AF4, MARKERS['YES','NO'], TICK, ContactQuality
# 1.170000000000000000e+02, 4.209743999999999687e+03, 4.235896999999999935e+03, 4.119487000000000080e+03, 4.166667000000000371e+03, 4.193332999999999629e+03, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+02
# 1.180000000000000000e+02, 4.206667000000000371e+03, 4.223590000000000146e+03, 4.116922999999999774e+03, 4.158974000000000160e+03, 4.196409999999999854e+03, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+02
# 1.190000000000000000e+02, 4.203077000000000226e+03, 4.221537999999999556e+03, 4.108204999999999927e+03, 4.151795000000000073e+03, 4.191282000000000153e+03, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+02
# 1.200000000000000000e+02, 4.200512999999999920e+03, 4.231795000000000073e+03, 4.104103000000000065e+03, 4.156922999999999774e+03, 4.183077000000000226e+03, 1.000000000000000000e+00, 1.000000000000000000e+00, 1.000000000000000000e+02

## MindBIGData ImageNet
# Multiple CSV files, each having the following entries
# AF3,4274.35897435897,4294.8717948718,4276.92307692308, ...
# AF4,4191.28205128205,4187.69230769231,4210.25641025641,4227.69230769231,...
# T7,4236.92307692308,4256.41025641026,4240.51282051282,4218.97435897436, ...
# T8,4351.79487179487,4350.25641025641,4318.46153846154,4305.64102564103,4319.48717948718,4335.89743589744 ...
# Pz,4189.74358974359,4172.30769230769,4146.66666666667,4140.51282051282,4157.4358974359,4185.64102564103, ...

# MindBIGData MNIST
# One .txt file, having the following format
# 1142043	173652	IN	AF3	0	256	4259.487179,4237.948717,4247.179487,4242.051282,4233.333333,4251.282051,4232.820512,4234.358974,4224.615384,4219.487179,4249.743589,4238.461538,4245.641025 ...
# 1142044	173652	IN	AF4	0	256	4103.076923,4100.512820,4102.564102,4087.692307,4074.358974,4095.897435 ...
# 1142045	173652	IN	T7	0	256	4245.128205,4218.461538,4242.051282,4245.128205,4233.333333,4257.435897,4241.025641,4241.538461,4231.282051,4230.256410,4261.538461,4233.333333,4237.435897,4250.256410 ...
# 1142046	173652	IN	T8	0	256	4208.717948,4188.717948,4204.102564,4198.461538,4179.487179,4203.589743,4194.871794,4185.128205,4174.358974,4183.589743,4208.717948,4172.820512,4185.128205,4200.000000,4168.717948,4184.615384,4179.487179,4182.564102,4182.564102,4169.230769,4196.410256,4181.025641,4188.205128...
# 1142047	173652	IN	PZ	0	256	4189.230769,4203.589743,4188.717948,4186.666666,4198.461538,4177.435897,4192.820512,4174.871794,4176.410256,4205.641025 ...

# Output Data : Formats
# Dict Object
# {"samples": torch.Tensor: Batch*Sequence*Channels}, "labels":torch.Tensor: Batch}
# Everything in float32 tensor format
# We use generic labels as placeholders

# Split: we will split data into train and valid sets
# We take a portion of imagenet data as validation set

import os
import torch
import argparse
import pandas as pd


def process_bdd_files(base_dir):
    dir_name = os.path.join(base_dir, "BDD")
    sub_dirs = [
        os.path.join(dir_name, "Baseline"),
        os.path.join(dir_name, "Image-Blank"),
        os.path.join(dir_name, "VisualQA"),
        os.path.join(dir_name, "Left-Right_Arrows"),
    ]
    data = []
    for dir in sub_dirs:
        files = os.listdir(dir)
        for fn in files:
            if "csv" in fn:
                handle = os.path.join(dir, fn)
                df = pd.read_csv(handle)
                af3 = df[" AF3"].tolist()[65 : 65 + 256]
                af4 = df[" AF4"].tolist()[65 : 65 + 256]
                pz = df[" Pz"].tolist()[65 : 65 + 256]
                t7 = df[" T7"].tolist()[65 : 65 + 256]
                t8 = df[" T8"].tolist()[65 : 65 + 256]
                individual_data = [af3, af4, pz, t7, t8]
                # individual_data = list(zip(*individual_data))
                data.append(individual_data)
    bdd_data = torch.tensor(data, dtype=torch.float32)
    print(f"Shape of data {bdd_data.shape}")
    return bdd_data


def process_mnist_files(base_dir):
    dir_name = os.path.join(base_dir, "MindBigData-IN-v1.06")
    handle = os.path.join(dir_name, "IN.txt")

    tmp_data = {}
    with open(handle) as f:
        for line in f:
            line = line.strip()
            line = line.split("\t")
            session = line[1]
            sess_data = tmp_data.get(session, [])
            sess_data.append([line[3].strip()] + line[6].split(","))
            tmp_data[session] = sess_data
    data = []
    for df in tmp_data.values():

        for d in df:
            if "AF3" in d[0]:
                af3 = list(map(float, d[1:129])) + [0] * (128 - len(d[1:129]))
                af3 = af3 + af3
            elif "AF4" in d[0]:
                af4 = list(map(float, d[1:129])) + [0] * (128 - len(d[1:129]))
                af4 = af4 + af4
            elif "PZ" in d[0]:
                pz = list(map(float, d[1:129])) + [0] * (128 - len(d[1:129]))
                pz = pz + pz
            elif "T7" in d[0]:
                t7 = list(map(float, d[1:129])) + [0] * (128 - len(d[1:129]))
                t7 = t7 + t7
            elif "T8" in d[0]:
                t8 = list(map(float, d[1:129])) + [0] * (128 - len(d[1:129]))
                t8 = t8 + t8

        individual_data = [af3, af4, pz, t7, t8]
        # individual_data = list(zip(*individual_data))
        data.append(individual_data)

    mnist_data = torch.tensor(data, dtype=torch.float32)
    print(f"Shape of data {mnist_data.shape}")
    return mnist_data


def process_imagenet_files(base_dir):
    dir_name = os.path.join(base_dir, "MindBigData-Imagenet")

    data = []
    files = os.listdir(dir_name)
    for fn in files:
        if "csv" in fn:
            handle = os.path.join(dir_name, fn)
            df = pd.read_csv(handle, header=None)
            for index, row in df.iterrows():
                # Convert the row to a list and append it to the list_of_lists
                d = row.tolist()

                if "AF3" in d[0]:
                    af3 = d[1:][65 : 65 + 256]
                elif "AF4" in d[0]:
                    af4 = d[1:][65 : 65 + 256]
                elif "Pz" in d[0]:
                    pz = d[1:][65 : 65 + 256]
                elif "T7" in d[0]:
                    t7 = d[1:][65 : 65 + 256]
                elif "T8" in d[0]:
                    t8 = d[1:][65 : 65 + 256]
            individual_data = [af3, af4, pz, t7, t8]
            # individual_data = list(zip(*individual_data))
            data.append(individual_data)
    imagenet_data = torch.tensor(data, dtype=torch.float32)
    print(f"Shape of data {imagenet_data.shape}")
    return imagenet_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prerpare Pretraining data")
    parser.add_argument("--dir", help="Path to the directory")
    args = parser.parse_args()
    bdd_data = process_bdd_files(args.dir)
    img_data = process_imagenet_files(args.dir)
    mnist_data = process_mnist_files(args.dir)

    # take a few samples from imagenet as validation split

    split_index = int(0.95 * len(img_data))

    imagenet_train_data = img_data[:split_index]
    valid_tensor = img_data[split_index:]

    total_data = torch.cat((bdd_data, imagenet_train_data, mnist_data), dim=0)
    
    mean_vals = total_data.mean(dim=[0,2])#, keepdim=True)[0].mean(dim=2, keepdim=True)[0]
    stdev_vals = total_data.std(dim=[0,2])# keepdim=True)[0].std(dim=2, keepdim=True)[0]

    mean_vals = mean_vals.reshape(1, 5, 1)
    stdev_vals = stdev_vals.reshape(1, 5, 1)

    #smoothing_factor = 1e-6

    normalized_dataset = (total_data - mean_vals) / stdev_vals #(max_vals - min_vals + smoothing_factor)
    print (f"mean_vals: {mean_vals}, stdev_vals: {stdev_vals}")
    print (normalized_dataset[0])

    torch.save({'mean_vals': mean_vals, 'stdev_vals': stdev_vals}, os.path.join(args.dir, "standard.pt"))

    print(f"Total training data {total_data.shape}")
    training_data = {
        "samples": total_data,
        "labels": torch.ones(total_data.size(0), dtype=torch.float32),
    }
    torch.save(training_data, os.path.join(args.dir, "train.pt"))

    print(f"Total test data {valid_tensor.shape}")
    test_data = {
        "samples": valid_tensor,
        "labels": torch.ones(valid_tensor.size(0), dtype=torch.float32),
    }
    torch.save(test_data, os.path.join(args.dir, "test.pt"))
    torch.save(test_data, os.path.join(args.dir, "val.pt"))
