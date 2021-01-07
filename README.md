# Hyper-Accelerated Tree Search (HATS)
This is the repo for the paper entitled 
"Towards Optimally Efficient Tree Search with Deep Temporal Difference Learning".
The key idea of this paper is to train the network towards the optimal heuristic. Please see more detials in the [arXiv preprint](https://www.arxiv.com).

## Requirements
To run the code you have to fullfill the following dependencies,
* `pip install matplotlib>=3.1`
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
