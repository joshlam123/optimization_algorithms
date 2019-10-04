# optimization_algorithms
Notebooks Implementing Different Optimization Algorithms

We want to solve hard optimization problems generally, and in a short amount of time without having to
develop tailored algorithms. There are hundreds of difierent problems and it would be too time-consuming
or be non-commercially valuable to work each problem specifically.

Enter Global Optima search methods such as Simulated Annealing and Genetic Algorithms, which have
been adapted to search an entire space of solutions. Here we explore Simulated and Population Annealing,
and Parallel Tempering with 2 classes of problems which can be scaled to solve many other similar problems.
A cooling schedule is an integral part of whether any of the algorithms are able to reach the global optima
quickly because of the acceptance criteria which accepts higher cost functions with a probability directly
proportional to the temperature

In here, are the implementations for Simulated Annealing, Population Annealing, Parallel Tempering, and Simulated Quantum Annealing for simple
TSP problem and also some classes of continuous problems taken from https://www.sfu.ca/~ssurjano/optimization.html
