
# Foreword
This is not a complete package, but a starting point to future research on asynchronous SGD based on caffe. You need some minimal efforts to make it work.

# Compile
- git clone --recursive git@github.com:raingo/caffe-mpi.git
- Compile ./caffe with CPU only
    - Has to be on a machine with CUDA SDK
- Install MPICH3
- Refer to ./mpi_env.csh to setup environment variables
- make

# Test
- Generate nodefile: Usually, you need to reserve nodes on your cluster. Edit nodefile to have the hostnames of the reserved machines. For example, if the torque system is used, you can use qsub to reserve machine and cat $PBS_NODEFILE to get the nodes reserved, and put it to the file nodefile
- ./release.sh: to pack everything to the distribute directory
    - You need to take a look at ./add-deps.sh as an example to resolve dependency on remote machines
- ./sync.sh: to copy everything onto cluster machines
- ./run_all_exp.sh: on the master machine (qsub) to run the experiments
- ./wrap-evaluate.sh: on a GPU machine to evaluate the trained model
    - Because there are intensive snapshot, a special format is used to store multiple snapshots. See snapshot.proto for details.
- ./wrap-plot.sh: generate the plots in the paper

# Source Files
- ./sgd-mpi.cpp: asynchronous SGD
- ./sgd.cpp: single node
- ./mpi.hpp: MPI primitives
- ./snapshot.proto: snapshots format for persistence
- ./flags.hpp: common flags used by ./sgd.cpp and ./sgd-mpi.cpp

# Contact
Please write your comments on the issue tracker

# Reference
Xiangru Lian, Yijun Huang, Yuncheng Li and Ji Liu. "Asynchronous Parallel Stochastic Gradient for Nonconvex Optimization." Advances in Neural Information Processing Systems (NIPS) 2015. [arxiv](http://arxiv.org/abs/1506.08272)
