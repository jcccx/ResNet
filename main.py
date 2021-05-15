from torch.utils.data import DataLoader
from tqdm import tqdm
from ResE import ResEncoder, BasicBlock, MyDataset
from sklearn.model_selection import train_test_split
import torch
import os
import csv
import pickle
import numpy as np
from random import sample
from collections import Counter
from torch.utils import data
import torch.utils.data as Data
from torch import nn, optim

import sys
import time
from IPython import display
from matplotlib import pyplot as plt

sys.path.append("..")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def use_svg_display():
    """Use svg format to display plot in jupyter"""
    display.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    # 设置图的尺寸
    plt.rcParams['figure.figsize'] = figsize

def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,
             legend=None, figsize=(50, 30)):
    set_figsize()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(x_vals, y_vals)
    #plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        plt.plot(x2_vals, y2_vals, linestyle=':')
        #plt.semilogy(x2_vals, y2_vals, linestyle=':')
        plt.legend(legend)
    plt.savefig(r"C:\Users\20180525\Desktop\loss.png", dpi = 300)
    plt.show()

def evaluate_accuracy(data_iter, net, loss, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定devicte就使用net的device
        device = list(net.parameters())[0].device
    test_ls_sum, acc_sum_1,  acc_sum_5, n, count = 0.0, 0.0, 0.0, 0, 0
    with torch.no_grad():
        for X, y in data_iter:
            maxk = max((1, 5))
            if isinstance(net, torch.nn.Module):
                net.eval()  # 评估模式, 这会关闭dropout
                out = net(X.to(device))
                _, pred = out.topk(maxk, 1, True, True)
                ls = loss(out, y.to(device))
                test_ls_sum += ls.cpu().item()
                acc_sum_1 += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                acc_sum_5 += (pred == y.to(device).view(-1, 1)).float().sum().cpu().item()
                net.train()  # 改回训练模式
            n += y.shape[0]
            count += 1
    return acc_sum_1 / n, acc_sum_5 / n, test_ls_sum/count

def ls_save(tpath, new_data):
    with open(tpath, 'a+') as f:
        f.write(str(new_data) + "\n")

def collet_label(labels, labels_counter):
    for k, (key, value) in enumerate(labels_counter):
        if (k != key):
            labels = [k if i == key else i for i in labels]
    #print(sorted(Counter(labels).items(), key=lambda x:x[0]))
    return labels

def train(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):

    net = net.to(device)
    print("training on ", device)
    loss = torch.nn.CrossEntropyLoss()
    train_ls, test_ls = [], []
    #for _ in tqdm(range(num_epochs), desc="epoch", leave=False):
    for epoch in range(num_epochs):
        #print(_)
        train_ls_sum, train_acc_sum_1, train_acc_sum_5, n, batch_count, start = 0.0, 0.0, 0.0, 0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            #print("y_hat:", y_hat.shape)
            #print("y:", y.shape)
            ls = loss(y_hat, y)

            optimizer.zero_grad()
            ls.backward()
            optimizer.step()
            train_ls_sum += ls.cpu().item()
            #print(ls.cpu().item())
            train_acc_sum_1 += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            _, y_pred = y_hat.topk(max((1, 5)), 1, True, True)
            train_acc_sum_5 += (y_pred == y.view(-1, 1)).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        train_l = train_ls_sum / batch_count
        test_acc_1, test_acc_5, test_l = evaluate_accuracy(test_iter, net, loss)
        train_ls.append(train_l)
        test_ls.append(test_l)
        ls_save(r"./train_ls.txt", train_l)
        ls_save(r"./test_ls.txt", test_l)
        #print("len(train_iter.dataset):", len(train_iter.dataset), "train n:", n)
        #print('epoch %d, train loss %.4f, test loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'% (epoch + 1, train_l, test_l, train_acc_sum / n, test_acc, time.time() - start))
        result = "epoch "+str(epoch+1)+", train loss "+str('%.4f' % train_l)+", test loss "+str('%.4f' % test_l)+\
            ', train acc 1 '+str('%.3f' % (train_acc_sum_1 / n))+', train acc 5 '+str('%.3f' % (train_acc_sum_5 / n))+\
                 ', test acc 1 '+str('%.3f' % test_acc_1)+', test acc 5 '+str('%.3f' % test_acc_5)+\
                 ', time '+str('%.1f' % (time.time() - start))+" sec"
        print(result)
        ls_save("./result.txt", result)
    semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
                 range(1, num_epochs + 1), test_ls, ['train', 'test'])

def dataset(root):
    users = os.listdir(root)
    users_path = [os.path.join(root, user) for user in users]
    user_apps = [os.listdir(user_path) for user_path in users_path]
    # print(user_apps)
    app_tlabel = []
    app_label1 = []
    app_label2 = []
    datas = []
    labels = []
    for apps in user_apps:
        for app in apps:
            # print(app)
            apptag = app[11:-5]
            app_path = os.path.join(root, app[:10] + app[-5:], app, app[:-5] + ".pkl")
            # print(app_path)
            fp = open(app_path, 'rb')
            if fp:
                # label
                if apptag not in app_tlabel:
                    app_tlabel.append(apptag)
                # label = torch.tensor([int(app_tlabel.index(apptag)), int(len(user_tlabel) - 1)]
                # feature
                while True:
                    try:
                        cc = pickle.load(fp)
                        # print(aa)
                        datas.append(np.array(cc).T)
                        labels.append(int(app_tlabel.index(apptag)))
                    except EOFError:
                        break
            fp.close()

    for i in range(len(app_tlabel)):
        c = labels.count(i)
        if (c <= 10):
            # del datas[labels.index(i)[0]]
            app_label1.append(app_tlabel[i])
            for j in range(labels.count(i)):
                del datas[labels.index(i)]
                labels.remove(i)
        # if (c > 400):
        #     app_label2.append(app_tlabel[i])
        #     locs = [index for (index, value) in enumerate(labels) if value == i]
        #     iloc = sample(locs, 400)
        #     loc = [x for x in locs if x not in iloc]
        #     for k in reversed(loc):
        #         del datas[k]
        #         del labels[k]

    #print(len(app_tlabel) - len(app_label1))
    labels_count = sorted(Counter(labels).items(), key=lambda x: x[0])
    #print(labels_count)
    print(app_label1)
    labels = collet_label(labels, labels_count)
    print(len(labels))
    print(Counter(labels))
    return datas, labels

def main():
    path = r"C:\Users\20180525\Desktop\compiler\sc_func_chunk"
    datas, labels = dataset(path)
    print("data[1]:",datas[1].shape)
    X_train, X_test, y_train, y_test = train_test_split(datas, labels, test_size=0.2, random_state=0,
                                                        stratify=labels)
    y_train_label = Counter(y_train)
    y_test_label = Counter(y_test)
    #print(y_train_label)
    #print(y_test_label)
    print('len(y_train_label):', len(y_train_label))
    print('len(y_test_label):', len(y_test_label))
    # print(sorted(y_train_label.items(), key=lambda x: x[0]))
    # print(sorted(y_test_label.items(), key=lambda x: x[0]))
    train_db = MyDataset(X_train, y_train)
    test_db = MyDataset(X_test, y_test)
    batch_size = 1024
    if sys.platform.startswith('win'):
        num_workers = 0  # 0表示不用额外的进程来加速读取数据
    else:
        num_workers = 4
    train_iter = torch.utils.data.DataLoader(train_db, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(test_db, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    # stage1:126(51) stage2:51
    net = ResEncoder(BasicBlock, [9, 7, 5, 3], 126, 88)
    lr, num_epochs = 0.003, 1000
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    train(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)


if __name__ == '__main__':
    main()