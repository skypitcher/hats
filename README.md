# Hyper-Accelerated Tree Search (HATS)
This is the repo for the paper entitled 
"Towards Optimally Efficient Tree Search with Deep Temporal Difference Learning".

This paper investigates the classical integer least-squares problem which estimates integer signals from linear models. The problem is NP-hard and often arises in diverse applications such as signal processing, bioinformatics, communications and machine learning, to name a few. Since the existing optimal search strategies involve prohibitive complexities, they are hard to be adopted in large-scale problems. To address this issue, we propose a general hyper-accelerated tree search (HATS) algorithm by employing a deep neural network to estimate the optimal heuristic for the underlying simplified memory-bounded A* algorithm, and the proposed algorithm can be easily generalized with other heuristic search algorithms. Inspired by the temporal difference learning, we further propose a training strategy which enables the network to approach the optimal heuristic precisely and consistently, thus the proposed algorithm can reach nearly the optimal efficiency when the estimation error is small enough. Experiments show that the proposed algorithm can reach almost the optimal maximum likelihood estimate performance in large-scale problems, with a very low complexity in both time and space. 

The key idea of this paper is to train the network towards the optimal heuristic. Please see more detials in [aXiv](https://arxiv.org/abs/2101.02420).

## Requirements
To run the code you have to fullfill the following dependencies,
* `matplotlib>=3.1`
* `tabulate>=0.8`
* `numpy>=1.16`
* `pytorch >= 1.5`
* `sortedcontainers>=2.3`

## Benchmark
You can change `n_ant` and `snr_list` and then run `python test.py` to see the result. 
Note that only `n_ant in [4, 8, 12, 16, 20, 24, 28, 32]` and `snr_list in range(5, 26)` are supported by the out-of-box, since we only train the model for these cases.


## Under Construction
TBD

## License
Anti 996 License Version 1.0

[![LICENSE](https://img.shields.io/badge/license-Anti%20996-blue.svg)](https://github.com/996icu/996.ICU/blob/master/LICENSE)
<a href="https://996.icu"><img src="https://img.shields.io/badge/link-996.icu-red.svg" alt="996.icu"></a>
