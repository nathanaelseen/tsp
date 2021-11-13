# tsp
An SLS (Stochastic Local Search) solver for TSP (Travelling Salesman Problem).

## Preface
This TSP solver started out as a school project for an NUS module (CS4234 - Optimisation Algorithms). Mainly for the assignment, we tackled this Kattis problem; [tsp](https://open.kattis.com/problems/tsp), trying to get as high a score possible.

I spent lots of effort into developing this TSP solver (about close to two months of intensive development), and didn't want to see my efforts go to waste. Thus, I've made my code public for anyone who might benefit from it. Feel free to send in PRs too, or fork this repository!

As of current development, the solver has reached state-of-the-art (SOTA) performance, scoring around 48.2/50 on average on Kattis!
<img width="1084" alt="Screenshot 2021-11-13 at 12 09 36 PM" src="https://user-images.githubusercontent.com/12388525/141605159-1de1403a-f24d-4203-9979-c64e1ca11ae9.png">

## Overview of TSP
TSP is an NP-hard problem, and for large number of vertices N, e.g; N = 1000, it near impossible to solve it via systematic search techniques such as brute-force, dynamic programming, as they incur exponential (non-polynomial) if not worse time-complexity.

To solve TSP for large graphs efficiently, we could use approximation algorithms such as [Christofides algorithm](https://en.wikipedia.org/wiki/Christofides_algorithm). The performance of these algorithms are easy to analyse however the problem is that they can provide solutions 100% off from the optimal tour.

In comparison, SLS works well in practice returning solutions which are near optimal most of the time, about 1-2% off the optimal. However, they are hard to analyse.

## Our TSP solver
We have chosen to use ILS (Iterated Local Search) as our SLS algorithm, as it has SOTA performance for TSP.
[<img width="573" alt="Screenshot 2021-11-13 at 1 43 55 PM" src="https://user-images.githubusercontent.com/12388525/141607405-75ccba12-ed64-4c08-8cf2-4c2f2805748d.png">](https://www.comp.nus.edu.sg/~stevenha/cs4234.html)

For our subsidiary local search, we used [2-opt](https://en.wikipedia.org/wiki/2-opt) and [3-opt](https://en.wikipedia.org/wiki/3-opt) hill-climb, taking first improving neighbour.

For perturbation, we used double-bridge moves and shuffling a random segment of the tour.

Finally, our acceptance criteria is simply to accept tours with equal or better cost than the current best found. If no such tour has been found from the current hill-climb iteration, we revert to best found tour.

## Future Work
The world record for this problem is currently at 48.81/50.
<img width="1120" alt="Screenshot 2021-11-13 at 1 53 04 PM" src="https://user-images.githubusercontent.com/12388525/141607626-17b5962e-58c4-4199-ab61-6d9ce576e661.png">

Submitting this code (``tsp.cpp``) verbatim gets you around 48/50 and (likely) also places you in the top 10 for this problem (I'm currently number 3!).

Generally, it's quite hard to beat this record or even improve from here. But my goal is to get as close to 50 as possible (or even 50), for this problem on Kattis! Anyone wants to join me in my quest?!

## Developement and Set-up
To set-up this project locally, here are the steps:
1. Clone the repository
2. Ensure you have C++ compiler installed, depending on your C++ compiler/flags, you may need to modify `GCC_ARGS` under `score.py` to use the correct compiler. For instance, since I'm using `g++-11` compiler I just put that compiler as the first item in the list.
3. Ensure you have Python3 installed.
4. Run `python3 score.py tsp.cpp`, and this will compile `tsp.cpp` and run the solver against our carefully curated test cases under the `/tests` folder (most of them are from [TSPLib](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/) and the rest are [VSLI](https://www.math.uwaterloo.ca/tsp/vlsi/index.html) and [national city](https://www.math.uwaterloo.ca/tsp/world/countries.html) test cases from the University of Waterloo), and score it similar to Kattis (i.e, upon 50).

Assuming you have set-up everything correctly, you should be able to see something like this;
<img width="1122" alt="Screenshot 2021-11-13 at 2 27 13 PM" src="https://user-images.githubusercontent.com/12388525/141608600-177541c2-c664-4404-9a49-4607a7efd8af.png">

On average, you would receive a score of around 47.5/50, which is consistently slightly lower than Kattis' test cases of 48/50.
