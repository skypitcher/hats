import time

from tabulate import tabulate

from model import HeuristicNet
from search import *
from prelude import *

import matplotlib.pyplot as plt


def get_heurisitc(n_ant):
    dnn = HeuristicNet(n_ant)
    dnn.load()
    return lambda y, R, psv: dnn.compute(y, R, psv)


def get_algorithms(n_ant, mem_size_list):
    alg_list = []
    hyber_accelerated_heuristic = get_heurisitc(n_ant)

    alg_list.append(["MMSE", mmse_estimate])

    for mem_size in mem_size_list:
        alg_list.append(
            (
                "SMA*({})".format(mem_size),
                SMAStar(capacity=mem_size, heuristic=None)
            )
        )
        alg_list.append(
            (
                "HATS({})".format(mem_size),
                SMAStar(capacity=mem_size, heuristic=hyber_accelerated_heuristic)
            )
        )

    return alg_list


def test_algorithms(algorithms, snr, n_ant, packet_length, total_packets):
    total_errs = np.zeros(len(algorithms))
    total_nodes_expanded_list = np.zeros(len(algorithms))
    total_nodes_generated_list = np.zeros(len(algorithms))

    last_len = 0
    for i_packet in range(total_packets):
        p = 10 ** (snr / 10)
        H = np.sqrt(p / n_ant) / np.sqrt(2) * complex_channel(n_ant)
        Q, R = np.linalg.qr(H)

        for i_timeslot in range(packet_length):
            b = random_bits([2 * n_ant, 1])
            x = qpsk(b)
            w = np.random.randn(2 * n_ant, 1)
            y = R @ x + Q.T @ w

            for i_alg, (alg_name, alg) in enumerate(algorithms):
                info_txt = "Testing n_ant={} snr={} packet={}/{} timeslots={}/{} algorithms={}/{} {}".format(
                    n_ant, snr, i_packet + 1, total_packets, i_timeslot + 1, packet_length, i_alg + 1, len(algorithms),
                    alg_name)

                if last_len > 0:
                    print(" " * last_len, end="\r")
                print(info_txt + "..", end="\r")
                last_len = len(info_txt) + 4

                x_est = alg(y, R)
                err = count_errors(x, x_est)
                total_errs[i_alg] += err
                total_nodes_expanded_list[
                    i_alg] += alg.nodes_expanded if "nodes_expanded" in alg.__dict__ is not None else 0
                total_nodes_generated_list[
                    i_alg] += alg.nodes_generated if "nodes_generated" in alg.__dict__ is not None else 0

        print()
        total_time_slots_now = (i_packet + 1) * packet_length
        total_bits_now = total_time_slots_now * 2 * n_ant

        precision = 1 / total_bits_now

        bers_list = total_errs / total_bits_now
        avg_nodes_expanded_list = total_nodes_expanded_list / total_time_slots_now
        avg_nodes_generated_list = total_nodes_generated_list / total_time_slots_now

        table = []
        for i_alg, (alg_name, alg) in enumerate(algorithms):
            table.append([
                alg_name,
                "{:e}({}/{})".format(bers_list[i_alg], total_errs[i_alg], total_bits_now),
                precision,
                "{:.2f}/{:.2f}".format(avg_nodes_expanded_list[i_alg], avg_nodes_generated_list[i_alg])
            ])
        print(tabulate(
            table,
            headers=["NAME", "BER", "PRECISION", "STEPS"],
            floatfmt=("", "", "e", ""),
            stralign="left",
            numalign="right"
        ))
        print()

    return bers_list, avg_nodes_expanded_list, avg_nodes_generated_list


def run_test(alg_list, snr, n_ant, packet_length, total_packets):
    t_start = time.time()
    results = test_algorithms(alg_list, snr, n_ant, packet_length, total_packets)
    t_end = time.time()
    t_total = t_end - t_start
    print("{:.2f} seconds elapsed".format(t_total))
    return results


def test_mimo_system(snr_list, n_ant, total_packets):
    omega = [-1, 1]
    m = 2 * n_ant
    u = m * len(omega)
    packet_length = 1024 // (2 * n_ant)
    mem_list = [u, u ** 2, np.inf]

    alg_list = get_algorithms(n_ant, mem_list)
    alg_ber_list = np.zeros([len(alg_list), len(snr_list)])
    alg_avg_nodes_generated_list = np.zeros([len(alg_list), len(snr_list)])

    fmt_list = ["k-+", "b-*", "r-o", "b--*", "r--o", "b:*", "r:o"]

    snr_results = []
    for snr in snr_list:
        snr_results.append(run_test(alg_list, snr, n_ant, packet_length, total_packets))

    for i_snr, snr in enumerate(snr_list):
        bers_list, avg_nodes_expanded_list, avg_nodes_generated_list = snr_results[i_snr]
        for i_alg in range(len(alg_list)):
            alg_ber_list[i_alg, i_snr] = bers_list[i_alg]
            alg_avg_nodes_generated_list[i_alg, i_snr] = avg_nodes_generated_list[i_alg]

    alg_complexity_coefficients = alg_avg_nodes_generated_list / m

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('QPSK {}x{} MIMO'.format(n_ant, n_ant))

    legend = []
    for i_alg, (alg_name, alg) in enumerate(alg_list):
        ax1.semilogy(snr_list, alg_ber_list[i_alg, :], fmt_list[i_alg])
        legend.append(alg_name)

    ax1.legend(legend)
    ax1.set_xlabel("SNR (dB)")
    ax1.set_ylabel("BER")
    ax1.set_xticks(snr_list)
    ax1.set_title("BER Performance Comparison")

    legend = []
    for i_alg, (alg_name, alg) in enumerate(alg_list):
        if alg_complexity_coefficients[i_alg, 0] > 0:
            ax2.semilogy(snr_list, alg_complexity_coefficients[i_alg, :], fmt_list[i_alg])
            legend.append(alg_name)

    ax2.legend(legend)
    ax2.set_xlabel("SNR (dB)")
    ax2.set_ylabel("Complexity coefficient")
    ax2.set_xticks(snr_list)
    ax2.set_title("Complexity Comparison")

    plt.show()


if __name__ == "__main__":
    test_mimo_system(snr_list=[13, 13.25, 13.5, 13.75, 14.25, 14.5, 14.75, 15], n_ant=32, total_packets=1000)
