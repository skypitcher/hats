import torch.nn as nn

from prelude import *


class RadiusNet(nn.Module):
    def __init__(self, n_ant: int):
        super(RadiusNet, self).__init__()
        self.n_ant = n_ant

        n = n_ant * 2
        self.layer_1 = nn.Linear(n + n * n, 128)
        self.layer_2 = nn.Linear(128, 64)
        self.layer_3 = nn.Linear(64, 32)
        self.layer_4 = nn.Linear(32, 32)
        self.layer_5 = nn.Linear(32, 16)
        self.layer_6 = nn.Linear(16, 1)

    def forward(self, y: torch.Tensor, R: torch.Tensor):
        y_ = torch_flat(y)
        R_ = torch_flat(R)
        x = torch.cat([y_, R_], dim=1)
        x = torch.relu(self.layer_1(x))
        x = torch.relu(self.layer_2(x))
        x = torch.relu(self.layer_3(x))
        x = torch.relu(self.layer_4(x))
        x = torch.relu(self.layer_5(x))
        x = torch.relu(self.layer_6(x))
        return x

    @property
    def name(self):
        return "qpsk_{}_nt{}".format(self.__class__.__name__, self.n_ant)

    def save(self):
        file_name = "saved_models/dlsd/{}.pkl".format(self.name)
        torch.save(self.state_dict(), file_name)
        print("{} saved".format(file_name))

    def load(self):
        file_name = "saved_models/dlsd/{}.pkl".format(self.name)
        state_dict = torch.load(file_name)
        # for k in state_dict:
        #     print(k)
        self.load_state_dict(state_dict)
        print("{} loaded".format(file_name))


class HeuristicNet(nn.Module):
    def __init__(self, n_ant: int):
        super(HeuristicNet, self).__init__()
        self.n_ant = n_ant

        n = n_ant * 2
        self.layer_1 = nn.Linear(3 * n + n * n, 128)
        self.layer_2 = nn.Linear(128, 64)
        self.layer_3 = nn.Linear(64, 32)
        self.layer_4 = nn.Linear(32, 32)
        self.layer_5 = nn.Linear(32, 16)

        self.layer_est = nn.Linear(16, n)
        self.layer_inf = nn.Linear(16, 1)

    def forward(self, y: torch.Tensor, R: torch.Tensor, psv: torch.Tensor):
        y_ = torch_flat(y)
        R_ = torch_flat(R)
        psv_ = torch_flat(psv)
        w_ = torch_flat(y - R @ psv)
        f = torch.cat([y_, R_, psv_, w_], dim=1)

        x = torch.relu(self.layer_1(f))
        x = torch.relu(self.layer_2(x))
        x = torch.relu(self.layer_3(x))
        x = torch.relu(self.layer_4(x))
        x = torch.relu(self.layer_5(x))

        s_ = torch.relu(self.layer_est(x))
        s_ = s_.unsqueeze(2)
        s_est = torch.sign(s_) / np.sqrt(2)
        s_est = torch.where(psv != 0, psv, s_est)
        w2 = torch.square(y - R @ s_est)
        w2[psv != 0] = 0
        h_est = torch.sum(w2, 1)

        h_inf = torch.relu(self.layer_inf(x))

        return torch.min(h_inf, h_est)

    def compute(self, y: np.ndarray, R: np.ndarray, psv: np.ndarray):
        y_ = torch.tensor(y, dtype=torch.float).unsqueeze(0)
        R_ = torch.tensor(R, dtype=torch.float).unsqueeze(0)
        psv_ = torch.tensor(psv, dtype=torch.float).unsqueeze(0)
        return self.forward(y_, R_, psv_).item()

    @property
    def name(self):
        return "qpsk_{}_nt{}".format(self.__class__.__name__, self.n_ant)

    def save(self):
        file_name = "saved_models/{}.pkl".format(self.name)
        torch.save(self.state_dict(), file_name)
        print("{} saved".format(file_name))

    def load(self):
        file_name = "saved_models/{}.pkl".format(self.name)
        state_dict = torch.load(file_name)
        # for k in state_dict:
        #     print(k)
        self.load_state_dict(state_dict)
        print("{} loaded".format(file_name))
