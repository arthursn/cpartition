# cpartition

Python library for modeling carbon redistribution by numerically solving Fick's laws of diffusion using the Finite Differences Method algorithm. The model supports fixed and mobile fcc/bcc interfaces. 

Different boundary conditions can be set depending on the taste of the user. The mixed-mode model is particularly useful for simulating scenarios where interface migration is not fully controlled by carbon diffusion.

Examples of use are simulation of kinetics of carbon redistribution during heat treatments, such as carburizing and the Quenching and Partitioning process.

Please refer to the following publications for detailed description of application and examples of utilization:

1. [A.S. Nishikawa, M.J. Santofimia, J. Sietsma, H. Goldenstein, Acta Mater. 142 (2018) 142-151.](https://dx.doi.org/10.1016/j.actamat.2017.09.048)

2. [A.S. Nishikawa, G. Miyamoto, T. Furuhara, A.P. Tschiptschin, H. Goldenstein, Acta Mater. 179 (2019) 1-16.](https://dx.doi.org/10.1016/j.actamat.2019.08.001)

# Installation and requirements

cpartition runs in python >= 3.5 using the following non-standard python libraries:

- numpy
- scipy
- matplotlib
- pandas
- periodictable 

First clone cpartition repository:

```bash
git clone https://github.com/arthursn/cpartition
```

Then install cpartition by running setup.py:

```bash
python3 setup.py install
```

Use the `--user` option to install cpartition in the user folder:

```bash
python3 setup.py install --user
```

Please notice that `setuptools` must be installed beforehand.

If cpartition is installed using `setup.py`, all dependencies should be automatically solved. Otherwise, the required libraries can be installed from the [Python Package Index](https://pypi.org) using pip:

```bash
pip3 install numpy scipy matplotlib pandas periodictable
```
