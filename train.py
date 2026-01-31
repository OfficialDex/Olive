# imports
import os
import json
import random
import threading
import time
import psutil
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from tinygrad.tensor import Tensor
from tinygrad.nn import Conv2d, Linear
from tinygrad.nn.optim import Adam
from tinygrad.nn.state import get_parameters

# variables
history_acc = []
history_loss = []
cpu_usage = 0.0
ram_usage = 0.0
running = True


def clearscreen():
    os.system("cls" if os.name == "nt" else "clear")


def systemmonitor():
    global cpu_usage, ram_usage, running
    p = psutil.Process(os.getpid())
    while running:
        cpu_usage = psutil.cpu_percent(interval=0.5)
        ram_usage = p.memory_info().rss / 1024 / 1024
        time.sleep(0.5)


def dashboard(e, total, acc, loss):
    clearscreen()
    print("training cnn (tinygrad)")
    print()
    print("epoch:", e, "/", total, f"({round(e/total*100,2)}%)")
    print("accuracy:", round(acc * 100, 2), "%")
    print("loss:", round(loss, 4))
    print()
    print("cpu usage:", round(cpu_usage, 2), "%")
    print("ram usage:", round(ram_usage, 2), "MB")


def loadimages():
    data = []
    for label in ["cat", "dog"]:
        folder = "dataset/" + label
        for f in os.listdir(folder):
            if f.lower().endswith("jpg"):
                data.append((folder + "/" + f, 0 if label == "cat" else 1))
    random.shuffle(data)
    return data


def loadimage(path):
    img = Image.open(path).resize((32, 32)).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return Tensor(arr).unsqueeze(0)


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


# training
def train(data, epochs):
    Tensor.training = True

    model = CNN()
    opt = Adam(get_parameters(model), lr=0.001)

    for e in range(1, epochs + 1):
        correct = 0
        loss_sum = 0

        batch = random.sample(data, min(256, len(data)))

        for path, y in batch:
            x = loadimage(path)
            t = Tensor([[y]])

            out = model(x)
            loss = (out - t).pow(2).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_sum += loss.item()
            if (out.item() > 0.5) == (y == 1):
                correct += 1

        acc = correct / len(batch)
        avg_loss = loss_sum / len(batch)

        history_acc.append(acc)
        history_loss.append(avg_loss)

        dashboard(e, epochs, acc, avg_loss)

    Tensor.training = False
    return model


def save(model):
    params = get_parameters(model)
    state = [p.numpy().tolist() for p in params]
    with open("olive.ai", "w") as f:
        json.dump(state, f)


def plot():
    plt.figure()
    plt.plot(history_acc, label="accuracy")
    plt.plot(history_loss, label="loss")
    plt.legend()
    plt.savefig("training.png")


# demo
def demo(model):
    print("\nfinal demo\n")
    for path, y in random.sample(loadimages(), 10):
        out = model(loadimage(path)).item()
        pred = "dog" if out > 0.5 else "cat"
        print("image:", path)
        print("prediction:", pred)
        print("confidence:", round(abs(out - 0.5) * 2, 3))
        print("correct:", "dog" if y else "cat")
        print()


if __name__ == "__main__":
    monitor = threading.Thread(target=systemmonitor, daemon=True)
    monitor.start()

    data = loadimages()
    print("images:", len(data))

    model = train(data, 20)

    running = False

    save(model)
    plot()
    demo(model)
