# GDMC for Deep Learning Weight Optimization — Final Report

This report aggregates the *short* sweep in `results/raw/*.csv`.

## Top-line conclusion

There is no universal-best optimizer; the right choice depends on the bit-width budget.

**The long runs overturn the short-run picture at 8 bits.** A 30-epoch re-run on MNIST MLP (full 60K training set) gives:

| optimizer | bits | best test acc |
|---|---|---|
| adam (continuous) | 32 | 0.9757 ± 0.0014 |
| **gdmc v2 (β1=0.9, k=1)** | **8** | **0.9775 ± 0.0005** |
| gdmc v2 (β1=0.9, k=1) | 4 | 0.9460 ± 0.0017 |
| gdmc v1 | 8 | 0.9608 ± 0.0004 |
| gdmc v1 | 4 | 0.8959 ± 0.0021 |
| projected-gd | 4 | 0.2180 ± 0.0763 |

**GDMC v2 at 8-bit matches/beats continuous Adam** (0.9775 vs 0.9757), and at 4-bit it is 3.0 points behind Adam while being 4.3× better than the QAT-style Projected-GD baseline. The short sweep below used only 2-3 epochs and understates GDMC substantially (GDMC 8-bit gains +0.186 from 3→30 epochs, the largest of any config). Full long-run analysis in `docs/long_runs.md`; per-cell best-optimizer tally in `docs/test_report.md` §3.7; v2 sweep in `docs/v2_results.md`.

## Headline table — best metric by (task, optimizer, bits)

*(short sweep only; long-run tasks are excluded and reported in `docs/long_runs.md`)*

### cifar10_cnn  (metric: best_test_acc)

| task        | model        | optimizer     | grid_spec   |   bits | mean_std (test_acc)        |    min |    max |   count |
|:------------|:-------------|:--------------|:------------|-------:|:----------------|-------:|-------:|--------:|
| cifar10_cnn | smallcnn_b16 | adam          | none        |     32 | 0.2513 ± 0.0034 | 0.2481 | 0.2549 |       3 |
| cifar10_cnn | smallcnn_b16 | gdmc          | uniform     |      4 | 0.2671 ± 0.0193 | 0.2470 | 0.2855 |       3 |
| cifar10_cnn | smallcnn_b16 | gdmc          | uniform     |      8 | 0.2775 ± 0.0061 | 0.2709 | 0.2828 |       3 |
| cifar10_cnn | smallcnn_b16 | gdmc-adaptive | uniform-pt  |      3 | 0.1439 ± 0.0135 | 0.1309 | 0.1578 |       3 |
| cifar10_cnn | smallcnn_b16 | gdmc-adaptive | uniform-pt  |      4 | 0.2702 ± 0.0151 | 0.2571 | 0.2868 |       3 |
| cifar10_cnn | smallcnn_b16 | gdmc-adaptive | uniform-pt  |      8 | 0.2698 ± 0.0144 | 0.2580 | 0.2859 |       3 |
| cifar10_cnn | smallcnn_b16 | momentum      | none        |     32 | 0.2766 ± 0.0304 | 0.2474 | 0.3080 |       3 |
| cifar10_cnn | smallcnn_b16 | projected-gd  | uniform     |      4 | 0.1208 ± 0.0157 | 0.1029 | 0.1319 |       3 |
| cifar10_cnn | smallcnn_b16 | projected-gd  | uniform     |      8 | 0.2727 ± 0.0302 | 0.2509 | 0.3071 |       3 |
| cifar10_cnn | smallcnn_b16 | sgd           | none        |     32 | 0.2206 ± 0.0042 | 0.2167 | 0.2250 |       3 |
| cifar10_cnn | smallcnn_b16 | sgld          | none        |     32 | 0.1091 ± 0.0027 | 0.1060 | 0.1112 |       3 |


### mnist_cnn  (metric: best_test_acc)

| task      | model        | optimizer     | grid_spec   |   bits | mean_std (test_acc)        |    min |    max |   count |
|:----------|:-------------|:--------------|:------------|-------:|:----------------|-------:|-------:|--------:|
| mnist_cnn | smallcnn_b16 | adam          | none        |     32 | 0.8041 ± 0.0934 | 0.6988 | 0.8767 |       3 |
| mnist_cnn | smallcnn_b16 | gdmc          | uniform     |      2 | 0.2133 ± 0.0704 | 0.1656 | 0.2942 |       3 |
| mnist_cnn | smallcnn_b16 | gdmc          | uniform     |      3 | 0.4574 ± 0.0628 | 0.4075 | 0.5279 |       3 |
| mnist_cnn | smallcnn_b16 | gdmc          | uniform     |      4 | 0.5180 ± 0.2193 | 0.1831 | 0.8640 |      18 |
| mnist_cnn | smallcnn_b16 | gdmc          | uniform     |      8 | 0.4617 ± 0.0247 | 0.4396 | 0.4883 |       3 |
| mnist_cnn | smallcnn_b16 | gdmc          | uniform     |     16 | 0.1250 ± 0.0504 | 0.0892 | 0.1826 |       3 |
| mnist_cnn | smallcnn_b16 | gdmc          | uniform     |     32 | 0.1210 ± 0.0435 | 0.0892 | 0.1706 |       3 |
| mnist_cnn | smallcnn_b16 | gdmc-adaptive | uniform-pt  |      3 | 0.2398 ± 0.0893 | 0.1507 | 0.3292 |       3 |
| mnist_cnn | smallcnn_b16 | gdmc-adaptive | uniform-pt  |      4 | 0.4018 ± 0.0314 | 0.3806 | 0.4378 |       3 |
| mnist_cnn | smallcnn_b16 | gdmc-adaptive | uniform-pt  |      8 | 0.3924 ± 0.0277 | 0.3630 | 0.4180 |       3 |
| mnist_cnn | smallcnn_b16 | momentum      | none        |     32 | 0.5559 ± 0.1245 | 0.4222 | 0.6686 |       3 |
| mnist_cnn | smallcnn_b16 | projected-gd  | uniform     |      2 | 0.1323 ± 0.0490 | 0.1033 | 0.1889 |       3 |
| mnist_cnn | smallcnn_b16 | projected-gd  | uniform     |      3 | 0.1430 ± 0.0479 | 0.1109 | 0.1980 |       3 |
| mnist_cnn | smallcnn_b16 | projected-gd  | uniform     |      4 | 0.1523 ± 0.0457 | 0.1122 | 0.2021 |       3 |
| mnist_cnn | smallcnn_b16 | projected-gd  | uniform     |      8 | 0.8708 ± 0.0407 | 0.8238 | 0.8961 |       3 |
| mnist_cnn | smallcnn_b16 | projected-gd  | uniform     |     16 | 0.8651 ± 0.0402 | 0.8416 | 0.9115 |       3 |
| mnist_cnn | smallcnn_b16 | projected-gd  | uniform     |     32 | 0.7210 ± 0.2185 | 0.4691 | 0.8595 |       3 |
| mnist_cnn | smallcnn_b16 | sgd           | none        |     32 | 0.2325 ± 0.0774 | 0.1569 | 0.3115 |       3 |
| mnist_cnn | smallcnn_b16 | sgld          | none        |     32 | 0.1060 ± 0.0071 | 0.1008 | 0.1141 |       3 |


### mnist_mlp  (metric: best_test_acc)

| task      | model       | optimizer     | grid_spec   |   bits | mean_std (test_acc)        |    min |    max |   count |
|:----------|:------------|:--------------|:------------|-------:|:----------------|-------:|-------:|--------:|
| mnist_mlp | mlp_256x256 | adam          | none        |     32 | 0.9325 ± 0.0019 | 0.9308 | 0.9345 |       3 |
| mnist_mlp | mlp_256x256 | gdmc          | uniform     |      2 | 0.7460 ± 0.0132 | 0.7334 | 0.7598 |       3 |
| mnist_mlp | mlp_256x256 | gdmc          | uniform     |      3 | 0.7776 ± 0.0064 | 0.7706 | 0.7831 |       3 |
| mnist_mlp | mlp_256x256 | gdmc          | uniform     |      4 | 0.8279 ± 0.0123 | 0.8137 | 0.8353 |       3 |
| mnist_mlp | mlp_256x256 | gdmc          | uniform     |      8 | 0.7745 ± 0.0065 | 0.7673 | 0.7798 |       3 |
| mnist_mlp | mlp_256x256 | gdmc          | uniform     |     16 | 0.0940 ± 0.0201 | 0.0709 | 0.1081 |       3 |
| mnist_mlp | mlp_256x256 | gdmc          | uniform     |     32 | 0.0896 ± 0.0226 | 0.0635 | 0.1028 |       3 |
| mnist_mlp | mlp_256x256 | gdmc-adaptive | uniform-pt  |      3 | 0.7319 ± 0.0166 | 0.7156 | 0.7487 |       3 |
| mnist_mlp | mlp_256x256 | gdmc-adaptive | uniform-pt  |      4 | 0.7366 ± 0.0101 | 0.7272 | 0.7472 |       3 |
| mnist_mlp | mlp_256x256 | gdmc-adaptive | uniform-pt  |      8 | 0.7366 ± 0.0100 | 0.7273 | 0.7472 |       3 |
| mnist_mlp | mlp_256x256 | momentum      | none        |     32 | 0.8566 ± 0.0039 | 0.8525 | 0.8602 |       3 |
| mnist_mlp | mlp_256x256 | projected-gd  | uniform     |      2 | 0.1987 ± 0.0872 | 0.1090 | 0.2831 |       3 |
| mnist_mlp | mlp_256x256 | projected-gd  | uniform     |      3 | 0.1984 ± 0.0862 | 0.1100 | 0.2822 |       3 |
| mnist_mlp | mlp_256x256 | projected-gd  | uniform     |      4 | 0.1997 ± 0.0878 | 0.1121 | 0.2877 |       3 |
| mnist_mlp | mlp_256x256 | projected-gd  | uniform     |      8 | 0.9284 ± 0.0147 | 0.9116 | 0.9387 |       3 |
| mnist_mlp | mlp_256x256 | projected-gd  | uniform     |     16 | 0.9389 ± 0.0032 | 0.9352 | 0.9409 |       3 |
| mnist_mlp | mlp_256x256 | projected-gd  | uniform     |     32 | 0.9298 ± 0.0044 | 0.9249 | 0.9336 |       3 |
| mnist_mlp | mlp_256x256 | sgd           | none        |     32 | 0.3839 ± 0.0700 | 0.3037 | 0.4326 |       3 |
| mnist_mlp | mlp_256x256 | sgld          | none        |     32 | 0.1110 ± 0.0049 | 0.1054 | 0.1144 |       3 |


### mnist_mlp_long_v2  (metric: best_test_acc)

| task              | model       | optimizer   | grid_spec   |   bits | mean_std (test_acc)        |    min |    max |   count |
|:------------------|:------------|:------------|:------------|-------:|:----------------|-------:|-------:|--------:|
| mnist_mlp_long_v2 | mlp_256x256 | gdmc        | uniform     |      4 | 0.8216 ± 0.1513 | 0.6022 | 0.9476 |       9 |
| mnist_mlp_long_v2 | mlp_256x256 | gdmc        | uniform     |      8 | 0.9702 ± 0.0075 | 0.9604 | 0.9780 |       9 |


### toy_regression  (metric: best_test_loss)

| task           | model     | optimizer     | grid_spec   |   bits | mean_std (test_loss)        |    min |     max |   count |
|:---------------|:----------|:--------------|:------------|-------:|:----------------|-------:|--------:|--------:|
| toy_regression | mlp_64x64 | adam          | none        |     32 | 0.0063 ± 0.0022 | 0.0043 |  0.0087 |       3 |
| toy_regression | mlp_64x64 | gdmc          | uniform     |      2 | 0.6721 ± 0.5537 | 0.3356 |  1.3112 |       3 |
| toy_regression | mlp_64x64 | gdmc          | uniform     |      3 | 0.3424 ± 0.0809 | 0.2530 |  0.4104 |       3 |
| toy_regression | mlp_64x64 | gdmc          | uniform     |      4 | 0.1454 ± 0.0189 | 0.1301 |  0.1665 |       3 |
| toy_regression | mlp_64x64 | gdmc          | uniform     |      8 | 0.0973 ± 0.0031 | 0.0954 |  0.1009 |       3 |
| toy_regression | mlp_64x64 | gdmc          | uniform     |     16 | 0.6060 ± 0.0339 | 0.5673 |  0.6308 |       3 |
| toy_regression | mlp_64x64 | gdmc          | uniform     |     32 | 0.6181 ± 0.0333 | 0.5798 |  0.6400 |       3 |
| toy_regression | mlp_64x64 | gdmc-adaptive | uniform-pt  |      3 | 0.1190 ± 0.0076 | 0.1142 |  0.1277 |       3 |
| toy_regression | mlp_64x64 | gdmc-adaptive | uniform-pt  |      4 | 0.1077 ± 0.0033 | 0.1052 |  0.1114 |       3 |
| toy_regression | mlp_64x64 | gdmc-adaptive | uniform-pt  |      8 | 0.1075 ± 0.0035 | 0.1049 |  0.1115 |       3 |
| toy_regression | mlp_64x64 | momentum      | none        |     32 | 0.0772 ± 0.0134 | 0.0691 |  0.0927 |       3 |
| toy_regression | mlp_64x64 | projected-gd  | uniform     |      2 | 9.2739 ± 4.8013 | 4.1973 | 13.7420 |       3 |
| toy_regression | mlp_64x64 | projected-gd  | uniform     |      3 | 0.8152 ± 0.0882 | 0.7328 |  0.9083 |       3 |
| toy_regression | mlp_64x64 | projected-gd  | uniform     |      4 | 0.5690 ± 0.0337 | 0.5313 |  0.5964 |       3 |
| toy_regression | mlp_64x64 | projected-gd  | uniform     |      8 | 0.0206 ± 0.0048 | 0.0151 |  0.0235 |       3 |
| toy_regression | mlp_64x64 | projected-gd  | uniform     |     16 | 0.0077 ± 0.0039 | 0.0032 |  0.0105 |       3 |
| toy_regression | mlp_64x64 | projected-gd  | uniform     |     32 | 0.0100 ± 0.0071 | 0.0033 |  0.0175 |       3 |
| toy_regression | mlp_64x64 | sgd           | none        |     32 | 0.1166 ± 0.0024 | 0.1143 |  0.1192 |       3 |
| toy_regression | mlp_64x64 | sgld          | none        |     32 | 0.2858 ± 0.2300 | 0.1521 |  0.5514 |       3 |


## Beta sweep (MNIST CNN, 4-bit)

### beta_sweep

| task      | model        | optimizer   | grid_spec   |   bits | mean_std (test_acc)        |    min |    max |   count |
|:----------|:-------------|:------------|:------------|-------:|:----------------|-------:|-------:|--------:|
| mnist_cnn | smallcnn_b16 | gdmc        | uniform     |      4 | 0.5210 ± 0.2097 | 0.1831 | 0.8640 |      15 |

