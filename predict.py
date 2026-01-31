# imports
import os
import json
import random
import time
import numpy as np
from PIL import Image
from tinygrad.tensor import Tensor
from tinygrad.nn import Conv2d, Linear
from tinygrad.nn.state import get_parameters


def loadimage(path):
    img = Image.open(path).resize((32, 32)).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return Tensor(arr).unsqueeze(0)


def loaddataset():
    data = []
    for label in ["cat", "dog"]:
        folder = "dataset/" + label
        for f in os.listdir(folder):
            if f.lower().endswith("jpg"):
                data.append((folder + "/" + f, label))
    return data


class CNN:
    def __init__(self):
        self.c1 = Conv2d(3, 8, 3)
        self.c2 = Conv2d(8, 16, 3)
        self.f1 = Linear(16 * 6 * 6, 32)
        self.f2 = Linear(32, 1)

    def __call__(self, x):
        x = self.c1(x).relu()
        x = x.max_pool2d(2)
        x = self.c2(x).relu()
        x = x.max_pool2d(2)
        x = x.flatten(1)
        x = self.f1(x).relu()
        x = self.f2(x).sigmoid()
        return x


def loadmodel():
    model = CNN()
    params = get_parameters(model)

    with open("olive.ai") as f:
        saved = json.load(f)

    for p, v in zip(params, saved):
        p.assign(np.array(v, dtype=np.float32))

    return model


def predict(model, path):
    start = time.time()
    out = model(loadimage(path)).item()
    duration = (time.time() - start) * 1000

    pred = "dog" if out > 0.5 else "cat"
    conf = abs(out - 0.5) * 2

    return pred, conf, round(duration, 2)


if __name__ == "__main__":
    Tensor.training = False

    model = loadmodel()
    dataset = loaddataset()

    samples = random.sample(dataset, 10)

    correct = 0

    print("\nrandom dataset predictions\n")

    for path, truth in samples:
        pred, conf, t = predict(model, path)

        if pred == truth:
            correct += 1

        print("image:", path)
        print("prediction:", pred)
        print("confidence:", round(conf, 3))
        print("correct:", truth)
        print("time:", t, "ms")
        print()

    print("dataset accuracy:", correct, "/", len(samples))
    print()

    print("unseen data test\n")

    for path, truth in [("doggy.jpeg", "dog"), ("catty.jpeg", "cat")]:
        pred, conf, t = predict(model, path)

        print("image:", path)
        print("prediction:", pred)
        print("confidence:", round(conf, 3))
        print("expected:", truth)
        print("time:", t, "ms")
        print()
