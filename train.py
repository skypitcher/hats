import random
import time

from prelude import *
from model import HeuristicNet
import torch
import torch.optim as optim
import torch.nn.functional as F


def gen_level(y, R, s, lv):
    cost = np.sum(np.square(y - R @ s))

    n = np.copy(s)
    n[:-lv] = 0
    g = tree_g(y, R, n)

    lv_par = lv - 1
    if lv_par == 0:
        par = np.zeros_like(n)
    else:
        par = np.copy(n)
        par[:-lv_par] = 0
    g_par = tree_g(y, R, par)
    par_mask = 0 if lv_par == 0 else 1

    lv_succ = lv + 1
    succ_1 = np.copy(n)
    succ_1[-lv_succ] = -1
    g_succ_1 = tree_g(y, R, succ_1)

    succ_2 = np.copy(n)
    succ_2[-lv_succ] = 1
    g_succ_2 = tree_g(y, R, succ_2)

    succ_mask = 0 if lv_succ == s.size else 1

    yield (
        np.expand_dims(y, 0),
        np.expand_dims(R, 0),
        np.reshape(cost, [1, 1]),
        np.expand_dims(n, 0),
        np.reshape(g, [1, 1]),
        np.expand_dims(par, 0),
        np.reshape(g_par, [1, 1]),
        np.reshape(par_mask, [1, 1]),
        np.expand_dims(succ_1, 0),
        np.reshape(g_succ_1, [1, 1]),
        np.expand_dims(succ_2, 0),
        np.reshape(g_succ_2, [1, 1]),
        np.reshape(succ_mask, [1, 1]),
    )


def gen_trajectory(y, R, s):
    data_set = []
    for lv in range(1, s.size):
        for data in gen_level(y, R, s, lv):
            data_set.append(data)
    return data_set


def random_data_set(snr_low, snr_high, n_ant: int, batch_size: int):
    data_set = []
    for i_batch in range(batch_size):
        snr = np.random.uniform(low=snr_low, high=snr_high)
        p = 10 ** (snr / 10)
        H = np.sqrt(p / n_ant) / np.sqrt(2) * complex_channel(n_ant)
        Q, R = np.linalg.qr(H)
        s = qpsk(random_bits([2 * n_ant, 1]))
        w = np.random.randn(2 * n_ant, 1)
        y = R @ s + Q.T @ w
        data_set.extend(gen_trajectory(y, R, s))
    random.shuffle(data_set)
    return data_set


def preproccess(data):
    data = [*zip(*data)]
    y = torch.from_numpy(np.concatenate(data[0])).float().cuda()
    R = torch.from_numpy(np.concatenate(data[1])).float().cuda()
    cost = torch.from_numpy(np.concatenate(data[2])).float().cuda()
    n = torch.from_numpy(np.concatenate(data[3])).float().cuda()
    g = torch.from_numpy(np.concatenate(data[4])).float().cuda()
    par = torch.from_numpy(np.concatenate(data[5])).float().cuda()
    g_par = torch.from_numpy(np.concatenate(data[6])).float().cuda()
    par_mask = torch.from_numpy(np.concatenate(data[7])).float().cuda()
    succ_1 = torch.from_numpy(np.concatenate(data[8])).float().cuda()
    g_succ_1 = torch.from_numpy(np.concatenate(data[9])).float().cuda()
    succ_2 = torch.from_numpy(np.concatenate(data[10])).float().cuda()
    g_succ_2 = torch.from_numpy(np.concatenate(data[11])).float().cuda()
    succ_mask = torch.from_numpy(np.concatenate(data[12])).float().cuda()

    return y, R, cost, n, g, par, g_par, par_mask, succ_1, g_succ_1, succ_2, g_succ_2, succ_mask


def compute_loss(model, target_model, train_set):
    y = train_set[0]
    R = train_set[1]
    cost = train_set[2]

    s = train_set[3]
    g = train_set[4]

    par = train_set[5]
    g_par = train_set[6]
    par_mask = train_set[7]

    succ_1 = train_set[8]
    g_succ_1 = train_set[9]
    succ_2 = train_set[10]
    g_succ_2 = train_set[11]
    succ_mask = train_set[12]

    h = model.forward(y, R, s)
    f = g + h

    h_par = target_model.forward(y, R, par).detach()
    f_par = g_par + h_par

    h_succ_1 = target_model.forward(y, R, succ_1).detach()
    expected_f_1 = g_succ_1 + h_succ_1 * succ_mask

    h_succ_2 = target_model.forward(y, R, succ_2).detach()
    expected_f_2 = g_succ_2 + h_succ_2 * succ_mask

    expected_f = torch.min(expected_f_1, expected_f_2)

    loss = torch.mean((f - expected_f) ** 2 + (f - f_par) ** 2 * par_mask + (f - cost) ** 2)

    return loss, torch.mean(cost), torch.mean(f), torch.mean(expected_f)


def run_train(snr_low, snr_high, n_ant, batch_size, batch_count, max_epoch):
    model = HeuristicNet(n_ant).cuda()
    try:
        model.load()
    except:
        pass

    print(model)

    target_model = HeuristicNet(n_ant).cuda()
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    optimizer = optim.Adam(model.parameters())

    i_epoch = 0
    while i_epoch < max_epoch:
        t_start = time.time()

        for i_batch in range(batch_count):
            data_set = random_data_set(snr_low, snr_high, n_ant, batch_size)
            train_set = preproccess(data_set)
            loss, a, b, c = compute_loss(model, target_model, train_set)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            text = "traning epoch={}".format(i_epoch + 1)
            text += " loss={:.4f}".format(loss.item())
            text += " {:.2f}/{:.2f}/{:.2f}".format(a.item(), b.item(), c.item())
            text += " batch={}/{}".format(i_batch + 1, batch_count)
            print(text)

            if (i_batch + 1) % 1 == 0:
                target_model.load_state_dict(model.state_dict())

        t_end = time.time()
        t_duration = (t_end - t_start) / 60
        print("{:.2f} mins elapsed".format(t_duration))

        model.save()

        i_epoch += 1


if __name__ == "__main__":
    run_train(snr_low=5, snr_high=26, n_ant=16, batch_size=200, batch_count=1000, max_epoch=1000)
