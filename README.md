# On Structured Prediction Theory with Calibrated Convex Surrogate Losses

This repo contains some MATLAB scripts providing numerical and symbolic computations of the calibration functions presented in our paper [On Structured Prediction Theory with Calibrated Convex Surrogate Losses](https://arxiv.org/abs/1703.02403).

The repo contains the following scripts:
* `calibrationFunc_01Loss_numerical.m` — numerical computation of the calibration functions for 0-1 loss (Proposition 10)
* `calibrationFunc_block01Loss_numerical.m` — numerical computation for the case of block 0-1 loss without constraints on the scores (Proposition 11)
* `calibrationFunc_block01Loss_constrScores_numerical.m` — numerical computation for the case of block 0-1 loss with constraints on the scores (Proposition 12)
* `calibrationFunc_hammingLoss_numerical.m` — numerical computation for the case of Hamming loss without constraints on the scores
* `calibrationFunc_hammingLoss_constrScores_numerical.m` — numerical computation for the case of Hamming loss with constraints on the scores (Proposition 13)
* `calibrationFunc_mixedLoss_numerical.m` — numerical computation for the case of mixed 0-1 and block 0-1 loss without constraints on the scores (Proposition 14)
* `calibrationFunc_mixedLoss_symbolic.m` — symbolic derivation helping to prove Proposition 14
* `calibrationFunc_mixedLoss_constrScores_numerical.m` — numerical computation for the case of mixed 0-1 and block 0-1 loss with constraints on the scores (Proposition 15)
* `calibrationFunc_mixedLoss_constrScores_symbolic.m` — symbolic derivation helping to prove Proposition 15

The scripts were tested on Ubuntu 16.04, Matlab-R2016a, but should run on other systems as well.
The numerical scripts depend on Optimization Toolbox, the symbolic scripts depend on Symbolic Math Toolbox.

The code is released under Apache v2 License allowing to use the code in any way you want.

### Citation

If you are using this software please cite the following paper in any resulting publication:
>@inproceedings{osokin17consistency,<br>
    title = {On Structured Prediction Theory with Calibrated Convex Surrogate Losses},<br>
    author = {Anton Osokin and Francis Bach and Simon Lacoste-Julien},<br>
    booktitle = {Advances in Neural Information Processing Systems (NIPS)},<br>
    year = {2017} }

### Authors

* [Anton Osokin](http://www.di.ens.fr/~osokin/)
* [Francis Bach](http://www.di.ens.fr/~fbach/)
* [Simon Lacoste-Julien](http://www.di.ens.fr/~slacoste/)
