# coding: utf-8

#import
import numpy as np
import matplotlib.pyplot as plt
import sys, os
import pickle
from collections import OrderedDict
from mnist import load_mnist

#function class
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]

class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]

class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7) 

class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)

        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)

        for key in params.keys():
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])

            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)
#main
def main1():
    from multi_layer_net import MultiLayerNet
    from util import smooth_curve

    # 0:MNISTデータの読み込み===
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)
    print(x_train.shape, t_train.shape, x_test.shape, t_test.shape)

    train_size = x_train.shape[0]
    batch_size = 128
    max_iterations = 2000

    # 1:実験の設定===
    optimizers = {}
    optimizers["SGD"] = SGD()
    optimizers["Momentum"] = Momentum()
    optimizers["AdaGrad"] = AdaGrad()
    optimizers["Adam"] = Adam()

    networks = {}
    train_loss = {}
    for key in optimizers.keys():
        networks[key] = MultiLayerNet(input_size=784,
                                      hidden_size_list=[100, 100, 100, 100],
                                      output_size=10)
        train_loss[key] = []


    # 2:訓練の開始===
    for i in range(max_iterations):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        for key in optimizers.keys():
            grads = networks[key].gradient(x_batch, t_batch)
            optimizers[key].update(networks[key].params, grads)

            loss = networks[key].loss(x_batch, t_batch)
            train_loss[key].append(loss)

        if i % 100 == 0:
            print("===" + "iteration:" + str(i) + "===")
            for key in optimizers.keys():
                loss = networks[key].loss(x_batch, t_batch)
                print(key + ":" + str(loss))

    # 3:グラフの描写===
    markers = {}
    mark = ["o", "x", "s", "D"]
    for key, i in zip(optimizers.keys(), range(len(optimizers))):
        markers[key] = mark[i]

    x = np.arange(max_iterations)
    for key in optimizers.keys():
        plt.plot(x, smooth_curve(train_loss[key]), marker=markers[key], markevery=100, label=key)
    plt.xlabel("iterations")
    plt.ylabel("loss")
    plt.ylim(0, 1)
    plt.legend()
    plt.show()

def main2():
    from multi_layer_net import MultiLayerNet
    from util import smooth_curve

    # 0:MNISTデータの読み込み===  
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

    train_size = x_train.shape[0]
    batch_size = 128
    max_iterations = 2000

    # 1:実験の設定===
    weight_init_types = {'std=0.01': 0.01, 'Xavier': 'sigmoid', 'He': 'relu'}
    optimizer = SGD(lr=0.01)

    networks = {}
    train_loss = {}
    for key, weight_type in weight_init_types.items():
        networks[key] = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100],
                                     output_size=10, weight_init_std=weight_type)
        train_loss[key] = []

    # 2:訓練の開始===
    for i in range(max_iterations):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        for key in weight_init_types.keys():
            grads = networks[key].gradient(x_batch, t_batch)
            optimizer.update(networks[key].params, grads)

            loss = networks[key].loss(x_batch, t_batch)
            train_loss[key].append(loss)

        if i % 100 == 0:
            print("===" + "iteration:" + str(i) + "===")
            for key in weight_init_types.keys():
                loss = networks[key].loss(x_batch, t_batch)
                print(key + ":" + str(loss))

    # 3:グラフの描写
    markers = {'std=0.01': 'o', 'Xavier': 's', 'He': 'D'}
    x = np.arange(max_iterations)
    for key in weight_init_types.keys():
        plt.plot(x, smooth_curve(train_loss[key]), marker=markers[key], markevery=100, label=key)
    plt.xlabel("iterations")
    plt.ylabel("loss")
    plt.ylim(0, 2.5)
    plt.legend()
    plt.show()


def main3():
    from multi_layer_net_extend import MultiLayerNetExtend
    from optimizer import SGD, Adam

    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

    # 学習データを削減
    x_train = x_train[:1000]
    t_train = t_train[:1000]

    max_epochs = 20
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.01

    def __train(weight_init_std):
        bn_network = MultiLayerNetExtend(input_size=784, hidden_size_list=[100, 100, 100, 100, 100],
                                         output_size=10, weight_init_std=weight_init_std, use_batchnorm=True)
        network = MultiLayerNetExtend(input_size=784, hidden_size_list=[100, 100, 100, 100, 100],
                                         output_size=10, weight_init_std=weight_init_std, use_batchnorm=False)

        optimizer = SGD(lr=learning_rate)

        train_acc_list = []
        bn_train_acc_list = []

        iter_per_epoch = max(train_size / batch_size, 1)
        epoch_cnt = 0

        for i in range(1000000000):
            batch_mask = np.random.choice(train_size, batch_size)
            x_batch = x_train[batch_mask]
            t_batch = t_train[batch_mask]

            for _network in (bn_network, network):
                grads = _network.gradient(x_batch, t_batch)
                optimizer.update(_network.params, grads)

            if i % iter_per_epoch == 0:
                train_acc = network.accuracy(x_train, t_train)
                bn_train_acc = bn_network.accuracy(x_train, t_train)
                train_acc_list.append(train_acc)
                bn_train_acc_list.append(bn_train_acc)

                print("epoch:" + str(epoch_cnt) + " | " + str(train_acc) + " - " + str(bn_train_acc))

                epoch_cnt += 1
                if epoch_cnt >= max_epochs:
                    break

        return train_acc_list, bn_train_acc_list

    # グラフの描写===
    weight_scale_list = np.logspace(0, -4, num=16)
    x = np.arange(max_epochs)

    for i, w in enumerate(weight_scale_list):
        print("===" + str(i+1) + "/16" + "===")
        train_acc_list, bn_train_acc_list = __train(w)

        plt.subplot(4,4,i+1)
        plt.title("W:" + str(w))
        if i == 15:
            plt.plot(x, bn_train_acc_list, label="Batch Normalization", markevery=2)
            plt.plot(x, train_acc_list, linestyle="--", label="Normal(without BatchNorm)", markevery=2)

        else:
            plt.plot(x, bn_train_acc_list, markevery=2)
            plt.plot(x, train_acc_list, linestyle="--", markevery=2)

        plt.ylim(0, 1.0)
        if i % 4:
            plt.yticks([])

        else:
            plt.ylabel("accuracy")

        if i < 12:
            plt.xticks([])

        else:
            plt.xlabel("epochs")

        plt.legend(loc="lower right")

    plt.show()


#main
def main():
    #main1()
    #main2()
    main3()
main()
