# Per-step training curves (archived)

The live curve directory (results/curves/) is excluded by .gitignore because
it is 124 MB of derived per-step CSVs. These archives preserve those curves
in git at about a third of the size.

| archive | source experiments | files | size | sha256 |
|---|---|---:|---:|---|
| mnist_mlp_v3.tar.gz | experiments/10 (GDMC v3, MNIST MLP) | 63 | 10 MB | c0787ee06afc173d2d03197fac99606104458eac05d1bc3470d326e7edc2cd8f |
| lowbit_mnist.tar.gz | experiments/15 (low-bit comparison, MNIST) | 160 | 11 MB | ebd38cf74d41fff55bbf129c83518eb32363e96d452b67f695aacd5cd14ae60b |
| lowbit_fashion.tar.gz | experiments/15 (low-bit comparison, Fashion-MNIST) | 128 | 9.7 MB | de70838f93c4430787ce384bb9372432d4eb53e37f8d2852ef9582e99ba92246 |
| fashion_mnist_mlp.tar.gz | experiments/11 (Fashion-MNIST) | 63 | 7.8 MB | 6c4459e70ac4b5a648119f33955683a04992f23c76a6d26fd6b71b9bb113a167 |

Each archive contains one directory named after the archive.

## Restore

    tar xzf results/curves_archive/lowbit_mnist.tar.gz -C results/curves

(create results/curves first if it does not exist)

## Columns

step, epoch, loss (minibatch train loss), test_loss, test_acc, accepted,
acceptance_rate, delta_loss, num_moved, accepted_step_size. Test metrics are
populated only on evaluation steps (every log_every steps).

## Regenerating instead

The curves are also written directly by the experiment scripts; for example:

    .venv/bin/python experiments/15_lowbit_comparison.py --task mnist --seeds 0 1 2 3 4 5 6 7 8 9
    .venv/bin/python analysis/lowbit_analysis.py

The learning-curve figures used in the write-ups are already committed under
results/plots/lowbit_learning_curves_mnist.png and
results/plots/lowbit_learning_curves_fashion.png.