# micropolis

A tiny Metropolis-Hastings neural networks optimizer, inspired by [micrograd](https://github.com/karpathy/micrograd).

This version is contemporary with [Micropolis II](https://smackcrane.github.io/notebook/micropolis-ii.html).

### References

* optim.MMH(test='Barker') implements the approximate Barker MH test of [6.3, [BDH17](https://www.jmlr.org/papers/volume18/15-205/15-205.pdf)].
* optim.MMH(test='exact') implements the exact Barker MH test of [[SPCC17](https://arxiv.org/abs/1610.06848)] (& thanks to [BIDData/BIDMach](https://github.com/BIDData/BIDMach/blob/master/data/MHTestCorrections/norm2log4000_20_1.0.txt) for the normal-to-logistic correction distriution).
