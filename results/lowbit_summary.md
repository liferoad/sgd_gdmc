# Same-precision low-bit comparison - summary

Runs: 288 from 2 task(s) and 10 distinct seeds. Settings were selected on a held-out validation split per method and bit-width (see results/lowbit_settings_*.json).

## Mean test accuracy (95% t-interval over seeds)

best_test_acc is the peak over the run; final_test_acc is the last
epoch. Reporting both matters because projected Adam peaks early and
then collapses at 2-4 bits.

| task    | method         |   bits | metric         |   n |   mean |    std |   ci_lo |   ci_hi |
|:--------|:---------------|-------:|:---------------|----:|-------:|-------:|--------:|--------:|
| fashion | gdmc-v1        |      2 | best_test_acc  |   8 | 0.6583 | 0.0163 |  0.6447 |  0.6720 |
| fashion | gdmc-v2        |      2 | best_test_acc  |   8 | 0.3274 | 0.0291 |  0.3030 |  0.3517 |
| fashion | projected-adam |      2 | best_test_acc  |   8 | 0.7351 | 0.0189 |  0.7193 |  0.7509 |
| fashion | projected-msgd |      2 | best_test_acc  |   8 | 0.4961 | 0.1258 |  0.3910 |  0.6013 |
| fashion | qat-ste-adam   |      2 | best_test_acc  |   8 | 0.7790 | 0.0190 |  0.7631 |  0.7949 |
| fashion | gdmc-v1        |      4 | best_test_acc  |   8 | 0.7631 | 0.0087 |  0.7558 |  0.7704 |
| fashion | gdmc-v2        |      4 | best_test_acc  |   8 | 0.7925 | 0.0098 |  0.7843 |  0.8007 |
| fashion | projected-adam |      4 | best_test_acc  |   8 | 0.7528 | 0.0192 |  0.7368 |  0.7688 |
| fashion | projected-msgd |      4 | best_test_acc  |   8 | 0.6363 | 0.0161 |  0.6229 |  0.6498 |
| fashion | qat-ste-adam   |      4 | best_test_acc  |   8 | 0.8798 | 0.0019 |  0.8782 |  0.8814 |
| fashion | gdmc-v1        |      8 | best_test_acc  |   8 | 0.8474 | 0.0011 |  0.8465 |  0.8484 |
| fashion | gdmc-v2        |      8 | best_test_acc  |   8 | 0.8751 | 0.0024 |  0.8731 |  0.8771 |
| fashion | projected-adam |      8 | best_test_acc  |   8 | 0.8666 | 0.0024 |  0.8646 |  0.8686 |
| fashion | projected-msgd |      8 | best_test_acc  |   8 | 0.0938 | 0.0192 |  0.0778 |  0.1098 |
| fashion | qat-ste-adam   |      8 | best_test_acc  |   8 | 0.8854 | 0.0027 |  0.8831 |  0.8876 |
| fashion | adam-fp32      |     32 | best_test_acc  |   8 | 0.8854 | 0.0022 |  0.8835 |  0.8872 |
| mnist   | gdmc-v1        |      2 | best_test_acc  |  10 | 0.7966 | 0.0056 |  0.7927 |  0.8006 |
| mnist   | gdmc-v2        |      2 | best_test_acc  |  10 | 0.4945 | 0.0601 |  0.4515 |  0.5374 |
| mnist   | projected-adam |      2 | best_test_acc  |  10 | 0.7960 | 0.1256 |  0.7062 |  0.8858 |
| mnist   | projected-msgd |      2 | best_test_acc  |  10 | 0.4094 | 0.1401 |  0.3092 |  0.5096 |
| mnist   | qat-ste-adam   |      2 | best_test_acc  |  10 | 0.9422 | 0.0030 |  0.9401 |  0.9444 |
| mnist   | gdmc-v1        |      4 | best_test_acc  |  10 | 0.8953 | 0.0032 |  0.8930 |  0.8976 |
| mnist   | gdmc-v2        |      4 | best_test_acc  |  10 | 0.9351 | 0.0062 |  0.9307 |  0.9396 |
| mnist   | projected-adam |      4 | best_test_acc  |  10 | 0.9170 | 0.0112 |  0.9090 |  0.9251 |
| mnist   | projected-msgd |      4 | best_test_acc  |  10 | 0.4306 | 0.1400 |  0.3304 |  0.5307 |
| mnist   | qat-ste-adam   |      4 | best_test_acc  |  10 | 0.9786 | 0.0015 |  0.9775 |  0.9796 |
| mnist   | gdmc-v1        |      8 | best_test_acc  |  10 | 0.9427 | 0.0015 |  0.9416 |  0.9438 |
| mnist   | gdmc-v2        |      8 | best_test_acc  |  10 | 0.9744 | 0.0010 |  0.9737 |  0.9751 |
| mnist   | projected-adam |      8 | best_test_acc  |  10 | 0.9739 | 0.0023 |  0.9723 |  0.9755 |
| mnist   | projected-msgd |      8 | best_test_acc  |  10 | 0.0947 | 0.0245 |  0.0772 |  0.1122 |
| mnist   | qat-ste-adam   |      8 | best_test_acc  |  10 | 0.9806 | 0.0012 |  0.9797 |  0.9814 |
| mnist   | adam-fp32      |     32 | best_test_acc  |  10 | 0.9801 | 0.0006 |  0.9797 |  0.9806 |
| fashion | gdmc-v1        |      2 | final_test_acc |   8 | 0.5566 | 0.0307 |  0.5309 |  0.5823 |
| fashion | gdmc-v2        |      2 | final_test_acc |   8 | 0.2923 | 0.0868 |  0.2198 |  0.3649 |
| fashion | projected-adam |      2 | final_test_acc |   8 | 0.3589 | 0.1169 |  0.2612 |  0.4567 |
| fashion | projected-msgd |      2 | final_test_acc |   8 | 0.4752 | 0.1196 |  0.3752 |  0.5752 |
| fashion | qat-ste-adam   |      2 | final_test_acc |   8 | 0.7572 | 0.0285 |  0.7333 |  0.7810 |
| fashion | gdmc-v1        |      4 | final_test_acc |   8 | 0.7542 | 0.0129 |  0.7434 |  0.7649 |
| fashion | gdmc-v2        |      4 | final_test_acc |   8 | 0.7507 | 0.0275 |  0.7277 |  0.7737 |
| fashion | projected-adam |      4 | final_test_acc |   8 | 0.5589 | 0.1530 |  0.4310 |  0.6868 |
| fashion | projected-msgd |      4 | final_test_acc |   8 | 0.6183 | 0.0274 |  0.5954 |  0.6412 |
| fashion | qat-ste-adam   |      4 | final_test_acc |   8 | 0.8790 | 0.0025 |  0.8769 |  0.8811 |
| fashion | gdmc-v1        |      8 | final_test_acc |   8 | 0.8473 | 0.0012 |  0.8463 |  0.8483 |
| fashion | gdmc-v2        |      8 | final_test_acc |   8 | 0.8751 | 0.0024 |  0.8731 |  0.8771 |
| fashion | projected-adam |      8 | final_test_acc |   8 | 0.8634 | 0.0040 |  0.8600 |  0.8667 |
| fashion | projected-msgd |      8 | final_test_acc |   8 | 0.0938 | 0.0192 |  0.0778 |  0.1098 |
| fashion | qat-ste-adam   |      8 | final_test_acc |   8 | 0.8831 | 0.0046 |  0.8793 |  0.8869 |
| fashion | adam-fp32      |     32 | final_test_acc |   8 | 0.8841 | 0.0026 |  0.8819 |  0.8863 |
| mnist   | gdmc-v1        |      2 | final_test_acc |  10 | 0.7768 | 0.0125 |  0.7679 |  0.7857 |
| mnist   | gdmc-v2        |      2 | final_test_acc |  10 | 0.4099 | 0.0656 |  0.3629 |  0.4568 |
| mnist   | projected-adam |      2 | final_test_acc |  10 | 0.4635 | 0.2278 |  0.3005 |  0.6264 |
| mnist   | projected-msgd |      2 | final_test_acc |  10 | 0.3853 | 0.1419 |  0.2837 |  0.4868 |
| mnist   | qat-ste-adam   |      2 | final_test_acc |  10 | 0.9354 | 0.0060 |  0.9311 |  0.9398 |
| mnist   | gdmc-v1        |      4 | final_test_acc |  10 | 0.8921 | 0.0041 |  0.8892 |  0.8951 |
| mnist   | gdmc-v2        |      4 | final_test_acc |  10 | 0.9335 | 0.0078 |  0.9280 |  0.9391 |
| mnist   | projected-adam |      4 | final_test_acc |  10 | 0.4907 | 0.2914 |  0.2822 |  0.6991 |
| mnist   | projected-msgd |      4 | final_test_acc |  10 | 0.4270 | 0.1401 |  0.3268 |  0.5273 |
| mnist   | qat-ste-adam   |      4 | final_test_acc |  10 | 0.9770 | 0.0018 |  0.9757 |  0.9783 |
| mnist   | gdmc-v1        |      8 | final_test_acc |  10 | 0.9426 | 0.0016 |  0.9414 |  0.9438 |
| mnist   | gdmc-v2        |      8 | final_test_acc |  10 | 0.9743 | 0.0010 |  0.9736 |  0.9750 |
| mnist   | projected-adam |      8 | final_test_acc |  10 | 0.9707 | 0.0030 |  0.9685 |  0.9728 |
| mnist   | projected-msgd |      8 | final_test_acc |  10 | 0.0947 | 0.0245 |  0.0772 |  0.1122 |
| mnist   | qat-ste-adam   |      8 | final_test_acc |  10 | 0.9768 | 0.0032 |  0.9745 |  0.9791 |
| mnist   | adam-fp32      |     32 | final_test_acc |  10 | 0.9791 | 0.0018 |  0.9779 |  0.9804 |

## Paired differences (matched by seed)

| task    |   bits | method         | metric         | comparison        |   n |   mean_diff |   median_diff |   paired_p |   wilcoxon_p |   n_positive |
|:--------|-------:|:---------------|:---------------|:------------------|----:|------------:|--------------:|-----------:|-------------:|-------------:|
| fashion |      2 | gdmc-v1        | best_test_acc  | vs FP32 Adam      |   8 |     -0.2270 |       -0.2297 |     0.0000 |     nan      |            0 |
| fashion |      2 | gdmc-v2        | best_test_acc  | vs FP32 Adam      |   8 |     -0.5580 |       -0.5530 |     0.0000 |     nan      |            0 |
| fashion |      2 | projected-adam | best_test_acc  | vs FP32 Adam      |   8 |     -0.1503 |       -0.1569 |     0.0000 |     nan      |            0 |
| fashion |      2 | projected-msgd | best_test_acc  | vs FP32 Adam      |   8 |     -0.3892 |       -0.3622 |     0.0001 |     nan      |            0 |
| fashion |      2 | qat-ste-adam   | best_test_acc  | vs FP32 Adam      |   8 |     -0.1064 |       -0.1081 |     0.0000 |     nan      |            0 |
| fashion |      2 | gdmc-v1        | best_test_acc  | vs projected-adam |   8 |     -0.0767 |       -0.0712 |     0.0000 |       0.0078 |            0 |
| fashion |      2 | gdmc-v2        | best_test_acc  | vs projected-adam |   8 |     -0.4077 |       -0.4054 |     0.0000 |       0.0078 |            0 |
| fashion |      2 | projected-msgd | best_test_acc  | vs projected-adam |   8 |     -0.2389 |       -0.2026 |     0.0012 |       0.0078 |            0 |
| fashion |      2 | qat-ste-adam   | best_test_acc  | vs projected-adam |   8 |      0.0439 |        0.0426 |     0.0058 |       0.0156 |            7 |
| fashion |      4 | gdmc-v1        | best_test_acc  | vs FP32 Adam      |   8 |     -0.1222 |       -0.1254 |     0.0000 |     nan      |            0 |
| fashion |      4 | gdmc-v2        | best_test_acc  | vs FP32 Adam      |   8 |     -0.0928 |       -0.0927 |     0.0000 |     nan      |            0 |
| fashion |      4 | projected-adam | best_test_acc  | vs FP32 Adam      |   8 |     -0.1326 |       -0.1225 |     0.0000 |     nan      |            0 |
| fashion |      4 | projected-msgd | best_test_acc  | vs FP32 Adam      |   8 |     -0.2490 |       -0.2493 |     0.0000 |     nan      |            0 |
| fashion |      4 | qat-ste-adam   | best_test_acc  | vs FP32 Adam      |   8 |     -0.0055 |       -0.0056 |     0.0005 |     nan      |            0 |
| fashion |      4 | gdmc-v1        | best_test_acc  | vs projected-adam |   8 |      0.0103 |       -0.0053 |     0.2620 |       0.7422 |            3 |
| fashion |      4 | gdmc-v2        | best_test_acc  | vs projected-adam |   8 |      0.0397 |        0.0371 |     0.0009 |       0.0078 |            8 |
| fashion |      4 | projected-msgd | best_test_acc  | vs projected-adam |   8 |     -0.1164 |       -0.1123 |     0.0000 |       0.0078 |            0 |
| fashion |      4 | qat-ste-adam   | best_test_acc  | vs projected-adam |   8 |      0.1270 |        0.1169 |     0.0000 |       0.0078 |            8 |
| fashion |      8 | gdmc-v1        | best_test_acc  | vs FP32 Adam      |   8 |     -0.0379 |       -0.0381 |     0.0000 |     nan      |            0 |
| fashion |      8 | gdmc-v2        | best_test_acc  | vs FP32 Adam      |   8 |     -0.0103 |       -0.0094 |     0.0001 |     nan      |            0 |
| fashion |      8 | projected-adam | best_test_acc  | vs FP32 Adam      |   8 |     -0.0188 |       -0.0180 |     0.0000 |     nan      |            0 |
| fashion |      8 | projected-msgd | best_test_acc  | vs FP32 Adam      |   8 |     -0.7916 |       -0.7926 |     0.0000 |     nan      |            0 |
| fashion |      8 | qat-ste-adam   | best_test_acc  | vs FP32 Adam      |   8 |     -0.0000 |       -0.0001 |     0.9819 |     nan      |            2 |
| fashion |      8 | gdmc-v1        | best_test_acc  | vs projected-adam |   8 |     -0.0192 |       -0.0189 |     0.0000 |       0.0078 |            0 |
| fashion |      8 | gdmc-v2        | best_test_acc  | vs projected-adam |   8 |      0.0085 |        0.0090 |     0.0002 |       0.0078 |            8 |
| fashion |      8 | projected-msgd | best_test_acc  | vs projected-adam |   8 |     -0.7728 |       -0.7729 |     0.0000 |       0.0078 |            0 |
| fashion |      8 | qat-ste-adam   | best_test_acc  | vs projected-adam |   8 |      0.0188 |        0.0180 |     0.0000 |       0.0078 |            8 |
| mnist   |      2 | gdmc-v1        | best_test_acc  | vs FP32 Adam      |  10 |     -0.1835 |       -0.1835 |     0.0000 |     nan      |            0 |
| mnist   |      2 | gdmc-v2        | best_test_acc  | vs FP32 Adam      |  10 |     -0.4857 |       -0.4875 |     0.0000 |     nan      |            0 |
| mnist   |      2 | projected-adam | best_test_acc  | vs FP32 Adam      |  10 |     -0.1841 |       -0.1171 |     0.0012 |     nan      |            0 |
| mnist   |      2 | projected-msgd | best_test_acc  | vs FP32 Adam      |  10 |     -0.5707 |       -0.5554 |     0.0000 |     nan      |            0 |
| mnist   |      2 | qat-ste-adam   | best_test_acc  | vs FP32 Adam      |  10 |     -0.0379 |       -0.0387 |     0.0000 |     nan      |            0 |
| mnist   |      2 | gdmc-v1        | best_test_acc  | vs projected-adam |  10 |      0.0007 |       -0.0631 |     0.9866 |       0.7695 |            3 |
| mnist   |      2 | gdmc-v2        | best_test_acc  | vs projected-adam |  10 |     -0.3015 |       -0.3455 |     0.0002 |       0.0039 |            1 |
| mnist   |      2 | projected-msgd | best_test_acc  | vs projected-adam |  10 |     -0.3866 |       -0.3796 |     0.0000 |       0.0020 |            0 |
| mnist   |      2 | qat-ste-adam   | best_test_acc  | vs projected-adam |  10 |      0.1462 |        0.0804 |     0.0049 |       0.0020 |           10 |
| mnist   |      4 | gdmc-v1        | best_test_acc  | vs FP32 Adam      |  10 |     -0.0848 |       -0.0848 |     0.0000 |     nan      |            0 |
| mnist   |      4 | gdmc-v2        | best_test_acc  | vs FP32 Adam      |  10 |     -0.0450 |       -0.0441 |     0.0000 |     nan      |            0 |
| mnist   |      4 | projected-adam | best_test_acc  | vs FP32 Adam      |  10 |     -0.0631 |       -0.0615 |     0.0000 |     nan      |            0 |
| mnist   |      4 | projected-msgd | best_test_acc  | vs FP32 Adam      |  10 |     -0.5496 |       -0.5084 |     0.0000 |     nan      |            0 |
| mnist   |      4 | qat-ste-adam   | best_test_acc  | vs FP32 Adam      |  10 |     -0.0015 |       -0.0020 |     0.0116 |     nan      |            2 |
| mnist   |      4 | gdmc-v1        | best_test_acc  | vs projected-adam |  10 |     -0.0217 |       -0.0232 |     0.0003 |       0.0039 |            1 |
| mnist   |      4 | gdmc-v2        | best_test_acc  | vs projected-adam |  10 |      0.0181 |        0.0146 |     0.0010 |       0.0020 |           10 |
| mnist   |      4 | projected-msgd | best_test_acc  | vs projected-adam |  10 |     -0.4865 |       -0.4527 |     0.0000 |       0.0020 |            0 |
| mnist   |      4 | qat-ste-adam   | best_test_acc  | vs projected-adam |  10 |      0.0615 |        0.0602 |     0.0000 |       0.0020 |           10 |
| mnist   |      8 | gdmc-v1        | best_test_acc  | vs FP32 Adam      |  10 |     -0.0374 |       -0.0380 |     0.0000 |     nan      |            0 |
| mnist   |      8 | gdmc-v2        | best_test_acc  | vs FP32 Adam      |  10 |     -0.0057 |       -0.0057 |     0.0000 |     nan      |            0 |
| mnist   |      8 | projected-adam | best_test_acc  | vs FP32 Adam      |  10 |     -0.0062 |       -0.0060 |     0.0000 |     nan      |            0 |
| mnist   |      8 | projected-msgd | best_test_acc  | vs FP32 Adam      |  10 |     -0.8854 |       -0.8800 |     0.0000 |     nan      |            0 |
| mnist   |      8 | qat-ste-adam   | best_test_acc  | vs FP32 Adam      |  10 |      0.0004 |        0.0004 |     0.3724 |     nan      |            6 |
| mnist   |      8 | gdmc-v1        | best_test_acc  | vs projected-adam |  10 |     -0.0312 |       -0.0321 |     0.0000 |       0.0020 |            0 |
| mnist   |      8 | gdmc-v2        | best_test_acc  | vs projected-adam |  10 |      0.0005 |       -0.0002 |     0.5704 |       0.4922 |            5 |
| mnist   |      8 | projected-msgd | best_test_acc  | vs projected-adam |  10 |     -0.8792 |       -0.8757 |     0.0000 |       0.0020 |            0 |
| mnist   |      8 | qat-ste-adam   | best_test_acc  | vs projected-adam |  10 |      0.0067 |        0.0072 |     0.0000 |       0.0020 |           10 |
| fashion |      2 | gdmc-v1        | final_test_acc | vs FP32 Adam      |   8 |     -0.3275 |       -0.3221 |     0.0000 |     nan      |            0 |
| fashion |      2 | gdmc-v2        | final_test_acc | vs FP32 Adam      |   8 |     -0.5917 |       -0.5648 |     0.0000 |     nan      |            0 |
| fashion |      2 | projected-adam | final_test_acc | vs FP32 Adam      |   8 |     -0.5251 |       -0.5303 |     0.0000 |     nan      |            0 |
| fashion |      2 | projected-msgd | final_test_acc | vs FP32 Adam      |   8 |     -0.4089 |       -0.3843 |     0.0000 |     nan      |            0 |
| fashion |      2 | qat-ste-adam   | final_test_acc | vs FP32 Adam      |   8 |     -0.1269 |       -0.1384 |     0.0000 |     nan      |            0 |
| fashion |      2 | gdmc-v1        | final_test_acc | vs projected-adam |   8 |      0.1977 |        0.1925 |     0.0013 |       0.0078 |            8 |
| fashion |      2 | gdmc-v2        | final_test_acc | vs projected-adam |   8 |     -0.0666 |       -0.0642 |     0.2315 |       0.3125 |            4 |
| fashion |      2 | projected-msgd | final_test_acc | vs projected-adam |   8 |      0.1162 |        0.1310 |     0.0941 |       0.0781 |            7 |
| fashion |      2 | qat-ste-adam   | final_test_acc | vs projected-adam |   8 |      0.3982 |        0.4127 |     0.0000 |       0.0078 |            8 |
| fashion |      4 | gdmc-v1        | final_test_acc | vs FP32 Adam      |   8 |     -0.1299 |       -0.1332 |     0.0000 |     nan      |            0 |
| fashion |      4 | gdmc-v2        | final_test_acc | vs FP32 Adam      |   8 |     -0.1334 |       -0.1325 |     0.0000 |     nan      |            0 |
| fashion |      4 | projected-adam | final_test_acc | vs FP32 Adam      |   8 |     -0.3252 |       -0.2908 |     0.0005 |     nan      |            0 |
| fashion |      4 | projected-msgd | final_test_acc | vs FP32 Adam      |   8 |     -0.2658 |       -0.2646 |     0.0000 |     nan      |            0 |
| fashion |      4 | qat-ste-adam   | final_test_acc | vs FP32 Adam      |   8 |     -0.0050 |       -0.0051 |     0.0072 |     nan      |            1 |
| fashion |      4 | gdmc-v1        | final_test_acc | vs projected-adam |   8 |      0.1953 |        0.1542 |     0.0117 |       0.0078 |            8 |
| fashion |      4 | gdmc-v2        | final_test_acc | vs projected-adam |   8 |      0.1918 |        0.1449 |     0.0081 |       0.0078 |            8 |
| fashion |      4 | projected-msgd | final_test_acc | vs projected-adam |   8 |      0.0594 |        0.0418 |     0.3592 |       0.4609 |            4 |
| fashion |      4 | qat-ste-adam   | final_test_acc | vs projected-adam |   8 |      0.3201 |        0.2851 |     0.0006 |       0.0078 |            8 |
| fashion |      8 | gdmc-v1        | final_test_acc | vs FP32 Adam      |   8 |     -0.0368 |       -0.0358 |     0.0000 |     nan      |            0 |
| fashion |      8 | gdmc-v2        | final_test_acc | vs FP32 Adam      |   8 |     -0.0090 |       -0.0082 |     0.0001 |     nan      |            0 |
| fashion |      8 | projected-adam | final_test_acc | vs FP32 Adam      |   8 |     -0.0207 |       -0.0206 |     0.0000 |     nan      |            0 |
| fashion |      8 | projected-msgd | final_test_acc | vs FP32 Adam      |   8 |     -0.7903 |       -0.7926 |     0.0000 |     nan      |            0 |
| fashion |      8 | qat-ste-adam   | final_test_acc | vs FP32 Adam      |   8 |     -0.0010 |       -0.0005 |     0.5512 |     nan      |            2 |
| fashion |      8 | gdmc-v1        | final_test_acc | vs projected-adam |   8 |     -0.0161 |       -0.0159 |     0.0000 |       0.0078 |            0 |
| fashion |      8 | gdmc-v2        | final_test_acc | vs projected-adam |   8 |      0.0117 |        0.0092 |     0.0006 |       0.0078 |            8 |
| fashion |      8 | projected-msgd | final_test_acc | vs projected-adam |   8 |     -0.7696 |       -0.7662 |     0.0000 |       0.0078 |            0 |
| fashion |      8 | qat-ste-adam   | final_test_acc | vs projected-adam |   8 |      0.0197 |        0.0214 |     0.0001 |       0.0078 |            8 |
| mnist   |      2 | gdmc-v1        | final_test_acc | vs FP32 Adam      |  10 |     -0.2024 |       -0.2037 |     0.0000 |     nan      |            0 |
| mnist   |      2 | gdmc-v2        | final_test_acc | vs FP32 Adam      |  10 |     -0.5693 |       -0.5766 |     0.0000 |     nan      |            0 |
| mnist   |      2 | projected-adam | final_test_acc | vs FP32 Adam      |  10 |     -0.5157 |       -0.5707 |     0.0001 |     nan      |            0 |
| mnist   |      2 | projected-msgd | final_test_acc | vs FP32 Adam      |  10 |     -0.5939 |       -0.6112 |     0.0000 |     nan      |            0 |
| mnist   |      2 | qat-ste-adam   | final_test_acc | vs FP32 Adam      |  10 |     -0.0437 |       -0.0456 |     0.0000 |     nan      |            0 |
| mnist   |      2 | gdmc-v1        | final_test_acc | vs projected-adam |  10 |      0.3133 |        0.3586 |     0.0016 |       0.0059 |            9 |
| mnist   |      2 | gdmc-v2        | final_test_acc | vs projected-adam |  10 |     -0.0536 |       -0.0301 |     0.5026 |       0.4922 |            4 |
| mnist   |      2 | projected-msgd | final_test_acc | vs projected-adam |  10 |     -0.0782 |       -0.0505 |     0.4532 |       0.5566 |            4 |
| mnist   |      2 | qat-ste-adam   | final_test_acc | vs projected-adam |  10 |      0.4720 |        0.5323 |     0.0001 |       0.0020 |           10 |
| mnist   |      4 | gdmc-v1        | final_test_acc | vs FP32 Adam      |  10 |     -0.0870 |       -0.0878 |     0.0000 |     nan      |            0 |
| mnist   |      4 | gdmc-v2        | final_test_acc | vs FP32 Adam      |  10 |     -0.0456 |       -0.0433 |     0.0000 |     nan      |            0 |
| mnist   |      4 | projected-adam | final_test_acc | vs FP32 Adam      |  10 |     -0.4885 |       -0.5037 |     0.0005 |     nan      |            0 |
| mnist   |      4 | projected-msgd | final_test_acc | vs FP32 Adam      |  10 |     -0.5521 |       -0.5141 |     0.0000 |     nan      |            0 |
| mnist   |      4 | qat-ste-adam   | final_test_acc | vs FP32 Adam      |  10 |     -0.0021 |       -0.0021 |     0.0295 |     nan      |            1 |
| mnist   |      4 | gdmc-v1        | final_test_acc | vs projected-adam |  10 |      0.4014 |        0.4191 |     0.0017 |       0.0098 |            8 |
| mnist   |      4 | gdmc-v2        | final_test_acc | vs projected-adam |  10 |      0.4429 |        0.4640 |     0.0010 |       0.0020 |           10 |
| mnist   |      4 | projected-msgd | final_test_acc | vs projected-adam |  10 |     -0.0636 |        0.0081 |     0.5285 |       0.7695 |            5 |
| mnist   |      4 | qat-ste-adam   | final_test_acc | vs projected-adam |  10 |      0.4864 |        0.5047 |     0.0005 |       0.0020 |           10 |
| mnist   |      8 | gdmc-v1        | final_test_acc | vs FP32 Adam      |  10 |     -0.0365 |       -0.0370 |     0.0000 |     nan      |            0 |
| mnist   |      8 | gdmc-v2        | final_test_acc | vs FP32 Adam      |  10 |     -0.0049 |       -0.0051 |     0.0000 |     nan      |            0 |
| mnist   |      8 | projected-adam | final_test_acc | vs FP32 Adam      |  10 |     -0.0085 |       -0.0084 |     0.0000 |     nan      |            0 |
| mnist   |      8 | projected-msgd | final_test_acc | vs FP32 Adam      |  10 |     -0.8844 |       -0.8795 |     0.0000 |     nan      |            0 |
| mnist   |      8 | qat-ste-adam   | final_test_acc | vs FP32 Adam      |  10 |     -0.0023 |       -0.0018 |     0.0744 |     nan      |            2 |
| mnist   |      8 | gdmc-v1        | final_test_acc | vs projected-adam |  10 |     -0.0281 |       -0.0290 |     0.0000 |       0.0020 |            0 |
| mnist   |      8 | gdmc-v2        | final_test_acc | vs projected-adam |  10 |      0.0036 |        0.0034 |     0.0122 |       0.0098 |            9 |
| mnist   |      8 | projected-msgd | final_test_acc | vs projected-adam |  10 |     -0.8760 |       -0.8714 |     0.0000 |       0.0020 |            0 |
| mnist   |      8 | qat-ste-adam   | final_test_acc | vs projected-adam |  10 |      0.0061 |        0.0062 |     0.0017 |       0.0059 |            9 |

Learning curves: results/plots/lowbit_learning_curves_fashion.png

Learning curves: results/plots/lowbit_learning_curves_mnist.png
