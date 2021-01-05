import time
from typing import Dict

from tabulate import tabulate

from model import HeuristicNet
from search import *
from prelude import *

import xlwt


class BERStat:
    def __init__(self):
        self.count = 0
        self.total_bits = 0
        self.prec = 0
        self.total_err = 0
        self.ber = 0
        self.total_nodes_generated = 0
        self.total_nodes_expanded = 0
        self.avg_nodes_generated = 0
        self.avg_nodes_expanded = 0
        self.max_mem_usage = 0

    def record(self, s, s_est, nodes_expanded, nodes_generated, maxmem):
        self.max_mem_usage = maxmem

        self.count += 1
        self.total_bits += s.size
        self.prec = 1 / self.total_bits
        self.total_err += count_errors(s, s_est)
        self.ber = self.total_err / self.total_bits

        self.total_nodes_expanded += nodes_expanded
        self.total_nodes_generated += nodes_generated
        self.avg_nodes_expanded = self.total_nodes_expanded / self.count
        self.avg_nodes_generated = self.total_nodes_generated / self.count

    def __repr__(self):
        return "BER={:e}({}/{})\tCOMP={:.2f}/{:.2f}\tPREC={:e}".format(
            self.ber, self.total_err, self.total_bits, self.avg_nodes_expanded, self.avg_nodes_generated, self.prec)


class BERTestTask:
    def __init__(self, name: str, detector):
        self.name = name
        self.detector = detector
        self.ber_stats: Dict[float, BERStat] = {}

    def run(self, snr: float, y: np.ndarray, R: np.ndarray, s: np.ndarray, omega):
        s_est = self.detector.search(y, R, omega)
        if snr not in self.ber_stats:
            self.ber_stats[snr] = BERStat()
        self.ber_stats[snr].record(
            s,
            s_est,
            self.detector.nodes_expanded,
            self.detector.nodes_generated,
            self.detector.max_mem_usage()
        )
        self._after_detect()

    def _after_detect(self):
        pass

    def info(self, snr):
        return "{}\t{}".format(self.name, self.ber_stats[snr])


class Test:
    def __init__(self):
        self.tasks: List[BERTestTask] = []

    def add_model(self, name, model):
        task = BERTestTask(name, model)
        self.tasks.append(task)
        print("Task {} created".format(task.name))

    def run(self, n_ant: int, snr: float, packet_length: int, total_packets: int):
        t_start = time.time()

        for i_packet in range(total_packets):
            p = 10 ** (snr / 10)
            H = np.sqrt(p / n_ant) / np.sqrt(2) * complex_channel(n_ant)
            Q, R = np.linalg.qr(H)

            for i_slot in range(packet_length):
                b = random_bits([2 * n_ant, 1])
                s = qpsk(b)
                w = np.random.randn(2 * n_ant, 1)
                y = R @ s + Q.T @ w

                for task in self.tasks:
                    task.run(snr, y, R, s, omega=[-1, 1])

            print("NUM_ANT={} SNR={} Packet={}/{}".format(n_ant, snr, i_packet + 1, total_packets))
            table = []
            for task in self.tasks:
                stats = task.ber_stats[snr]
                table.append([
                    task.name,
                    "{:e}({}/{})".format(stats.ber, stats.total_err, stats.total_bits),
                    stats.prec,
                    "{:.2f}/{:.2f}".format(stats.avg_nodes_expanded, stats.avg_nodes_generated),
                    stats.max_mem_usage
                ])
            print(tabulate(
                table,
                headers=["Name", "BER", "PREC", "Complexity", "MaxMemUsage"],
                floatfmt=("", "", "e", "", ""),
                stralign="left",
                numalign="right"
            ))
            print()

        t_end = time.time()
        t_total = t_end - t_start
        print("{:.2f} seconds elapsed".format(t_total))

    def save_xls(self, filename):
        workbook = xlwt.Workbook(encoding='utf-8')
        sheet = workbook.add_sheet('benchmark')

        snr_ok = False
        offset = len(self.tasks) + 1
        sheet.write(0, 0, "SNR")
        for i_task, task in enumerate(self.tasks):
            sheet.write(0, i_task + 1, task.name)
            sheet.write(0, offset + i_task + 1, task.name)

            if not snr_ok:
                for i_snr, snr in enumerate(task.ber_stats):
                    sheet.write(i_snr + 1, 0, "{:02}".format(snr))
                snr_ok = True

            for i_snr, snr in enumerate(task.ber_stats):
                sheet.write(i_snr + 1, i_task + 1, "{:e}".format(task.ber_stats[snr].ber))
                sheet.write(i_snr + 1, offset + i_task + 1, "{:.2f}".format(task.ber_stats[snr].avg_nodes_generated))

        workbook.save(filename)


def get_heurisitc(n_ant):
    dnn = HeuristicNet(n_ant)
    dnn.load()
    return lambda y, R, psv: dnn.compute(y, R, psv)


def test_all():
    test = Test()

    n_ant = 12
    packet_length = 1024 // (2 * n_ant)
    total_packets = 9999999

    hyber_accelerated_heuristic = get_heurisitc(n_ant)

    omega = [-1, 1]
    mem_unit = 2 * n_ant * len(omega)
    mem_size_list = [mem_unit, mem_unit ** 2, np.inf]
    # mem_size_list = [np.inf]

    for mem_size in mem_size_list:
        test.add_model("SMA*({})".format(mem_size), SMAStar(capacity=mem_size, heuristic=None))

    for mem_size in mem_size_list:
        test.add_model("HATS({})".format(mem_size), SMAStar(capacity=mem_size, heuristic=hyber_accelerated_heuristic))

    for snr in [25]:
        test.run(n_ant, snr, packet_length, total_packets)

    test.save_xls("result/temp.xls")


if __name__ == "__main__":
    test_all()
