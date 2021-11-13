/*
 * Acknowledgement:
 * In our 48++/50 submission for tsp, we adapted code/ideas from various online GitHub
 * repositories that contain solutions kattis tsp too.
 *
 * 1. Adapated code from this repository:
 *    https://github.com/estan/tsp
 *
 * 2. Adapated code from one of the previous top CS4234 teams on TSP:
 *   https://github.com/Lookuz/CS4234-Stochastic-Local-Search-Methods/tree/master/TSP
 */
#include <algorithm>
#include <iostream>
#include <limits>
#include <random>
#include <queue>
#include <cassert>
#include <chrono>
#include <tuple>

using namespace std;

typedef uint_fast64_t ll;
typedef pair<double, double> dd;
typedef vector<dd> vdd;
typedef vector<uint_fast16_t> vi;
typedef vector<double> vd;
typedef pair<uint_fast32_t, uint_fast16_t> ii;
typedef vector<ii> vii;

// Hyperparameters for the SLS algorithm
const static uint_fast16_t NEIGH_LIMIT = 20;
const static uint_fast16_t GEO_SHUFFLE_WIDTH = 30;
const static uint_fast16_t SHUFFLE_PROBABILITY = 24; // chance to shuffle vs double bridge, value from 0 to 100
const static uint_fast16_t EXECUTION_DURATION = 1997; // time for the whole algo, give 5ms to cout best tour
const static uint_fast16_t THREE_OPT_BUFFER_TIME = 15;

// Set-up RNG
random_device rd;
mt19937 gen(rd());

/**
 * A very simple N x M matrix class.
 */
template<typename T>
class Matrix {
public:
    Matrix(std::size_t N, std::size_t M) :
        m_data(N * M, T()),
        m_rows(N),
        m_cols(M) {}

    inline T* operator[](int i) {
        return &m_data[i * m_cols];
    }

    inline T const* operator[](int i) const {
        return &m_data[i * m_cols];
    }

    inline std::size_t getNumRows() const {
        return m_rows;
    }

    inline std::size_t getNumCols() const {
        return m_cols;
    }

private:
    std::vector<T> m_data;
    std::size_t m_rows;
    std::size_t m_cols;
};

typedef Matrix<uint_fast16_t> mi16;
typedef Matrix<uint_fast32_t> mi32;

// Return the current time.
static inline chrono::time_point<chrono::high_resolution_clock> now() {
    return chrono::high_resolution_clock::now();
}

// Fills up the distance matrix as per the problem specs as euclidean distance
// rounded to the nearest integer. Takes `n` number of 2D `coords` as input.
mi32 create_dist_matrix(istream& cin) {
    size_t n; cin >> n;

    vd x(n);
    vd y(n);
    for (uint_fast16_t i = 0; i < n; ++i) {
        cin >> x[i] >> y[i];
    }

    mi32 dist(n, n);

    for (uint_fast16_t i = 0; i < n - 1; ++i) {
        for (uint_fast16_t j = i + 1; j < n; ++j) {
            // Euclidean distance, rounded as per problem specs
            dist[i][j] = dist[j][i] = round(sqrt(pow(x[i] - x[j], 2) + pow(y[i] - y[j], 2)));
        }
    }

    return dist;
}

// Fills up nearest neighbour matrix where for each node i, we compute the nearest
// neighbour. This method has to be called after fill_up_dist_matrix populates the
// dist[MAX_n][MAX_n] array
mi16 create_neighbour_matrix(const mi32& dist, uint_fast16_t k) {
    uint_fast16_t n = dist.getNumRows();
    uint_fast16_t m = dist.getNumCols() - 1; // node is not neighbor of itself
    k = min(m, k);
    mi16 neighbour(n, k);

    priority_queue<ii, vii, greater<ii>> pq;

    uint_fast16_t j = 0;
    uint_fast16_t d, v;
    for (uint_fast16_t i = 0; i < n; ++i) {
        for (j = 0; j < n; ++j) {
            /*
             * This is important, we don't want a node to become it's own nearest
             * neighbour
             */
            if (i == j) continue;

            pq.push({dist[i][j], j});
        }

        j = 0;
        while (j < k) {
            tie(d, v) = pq.top(); pq.pop();

            neighbour[i][j] = v;
            j++;
        }

        pq = priority_queue<ii, vii, greater<ii>>();
    }

    return neighbour;
}

// Computes the cost of a given tour, takes O(n)
inline ll compute_cost(const vi& tour, const mi32& dist) {
    size_t n = tour.size();

    ll cost = 0;
    for (uint_fast16_t i = 0, j = 1; i < n; ++i, ++j) {
        cost += dist[tour[i]][tour[j % n]];
    }

    return cost;
}

inline void swap_adj_seg(vi& tour, uint_fast16_t* position, uint_fast16_t A, uint_fast16_t B, uint_fast16_t C, uint_fast16_t D) {
    size_t n = tour.size();

    vi temp;

    uint_fast16_t cur = C;
    while (cur != D) {
        temp.push_back(tour[cur]);
        cur = (cur + 1) % n;
    }
    temp.push_back(tour[cur]); // temp contains segment [C .. D]

    cur = A;
    while (cur != B) {
        temp.push_back(tour[cur]);
        cur = (cur + 1) % n;
    }
    temp.push_back(tour[cur]); // temp contains segment [C .. D, A .. B]

    for (uint_fast16_t i = 0; i < temp.size(); ++i) { // copy over to tour
        tour[(A + i) % n] = temp[i];
        position[temp[i]] = (A + i) % n;
    }
}

// Prints a given tour (linear fashion/order)
void print_tour(const vi& tour) {
    for (auto v : tour) {
        cout << v << "\n";
    }
}

// Gives a 2-approximate solution to TSP based on the greedy nearest neighbour
// approach. Used as the starting point for local search.
vi naive_algo(const mi32& dist) {
    size_t n = dist.getNumRows();

    bool used[n];
    for (uint_fast16_t i = 0; i < n; ++i) {
        used[i] = false;
    }

    vi tour(n);
    tour[0] = 0;
    used[0] = true;

    for (uint_fast16_t i = 1; i < n; ++i) {
        int best = -1;
        for (uint_fast16_t j = 0; j < n; ++j) {
            if (!used[j] && (best == -1 || dist[tour[i-1]][j] < dist[tour[i-1]][best])) {
                best = j;
            }
        }
        tour[i] = best;
        used[best] = true;
    }

    return tour;
}

// Copied reverse method from: https://github.com/estan/tsp/blob/master/tsp.cpp
inline void reverse_tour(vi& tour, uint_fast16_t start, uint_fast16_t end, uint_fast16_t* position) {
    size_t n = tour.size();

    uint_fast16_t numSwaps = (((start <= end ? end - start : (end + n) - start) + 1) / 2);
    uint_fast16_t i = start;
    uint_fast16_t j = end;
    while (numSwaps--) {
        swap(tour[i], tour[j]);

        position[tour[i]] = i;
        position[tour[j]] = j;

        i = (i + 1) % n;
        j = ((j + n) - 1) % n;
    }
}


// This randomly selects a subtour and shuffles it
// modifies tour in place
inline void shuffle_tour(vi& tour) {
    size_t n = tour.size();

    if (n <= GEO_SHUFFLE_WIDTH) {
        shuffle(tour.begin(), tour.end(), rd);
        return;
    }
    uniform_int_distribution<size_t> randomOffset(0, tour.size() - GEO_SHUFFLE_WIDTH); // [a, b] Inclusive
    int left = randomOffset(rd);
    // cout << "left: " << left << "\n";
    shuffle(tour.begin() + left, tour.begin() + left + GEO_SHUFFLE_WIDTH, rd); // [a, b) exclusive
    return;
}

// Performs a double bridge perturbation on the tour
inline vi double_bridge(vi& tour) {
    size_t n = tour.size();

    vi newTour;
    newTour.reserve(n);

    if (n < 8) {
        newTour = tour;
        shuffle(newTour.begin(), newTour.end(), rd);
        return newTour;
    }

    uniform_int_distribution<uint_fast16_t> randomOffset(1, n / 4);
    uint_fast16_t A = randomOffset(rd);
    uint_fast16_t B = A + randomOffset(rd);
    uint_fast16_t C = B + randomOffset(rd);
    copy(tour.begin(), tour.begin() + A, back_inserter(newTour));
    copy(tour.begin() + C, tour.end(), back_inserter(newTour));
    copy(tour.begin() + B, tour.begin() + C, back_inserter(newTour));
    copy(tour.begin() + A, tour.begin() + B, back_inserter(newTour));
    return newTour;
}

// Performs a 2-opt move on a given tour, where we select 2 distinct edges based on
// nearest neighbour heuristic (and take the first improving perturbative neighbour)
// Terminates when there are no improving neighbours or when time limit exeeded.
inline void two_opt(vi& tour, const mi32& dist, const mi16& neighbour, uint_fast16_t* position) {
    uint_fast16_t n = tour.size();

    uint_fast16_t u1, u2, u3, u4;
    uint_fast16_t u1Idx, u2Idx, u3Idx, u4Idx;

    bool improvingTSP = true; // Exit when no improving neighbours

    /*
     * Exit SLS if no improving TSP (neighbours), i.e, we can't optimise the TSP any
     * further
     */
    while (improvingTSP) {
        improvingTSP = false;

        for (u1Idx = 0; u1Idx < n; ++u1Idx) {
            u1 = tour[u1Idx];
            u2Idx = (u1Idx + 1) % n;
            u2 = tour[u2Idx];

            for (uint_fast16_t k = 0; k < neighbour.getNumCols(); ++k) {
                u3Idx = position[neighbour[u1][k]];
                u3 = tour[u3Idx];

                // All subsequent neighbours of u1 guaranteed to have longer distances
                if (dist[u1][u3] >= dist[u1][u2]) break;

                u4Idx = (u3Idx + 1) % n;
                u4 = tour[u4Idx];

                if (u1 == u4 || u2 == u3) continue;

                // Using technique for fast w(p') computation from lecture slides
                if (dist[u1][u3] + dist[u2][u4] < dist[u1][u2] + dist[u3][u4]) {
                    improvingTSP = true;

                    reverse_tour(tour, u2Idx, u3Idx, position);

                    break;
                }
            }
        }
    }
}

// Performs a 3-opt move on a given tour, where we select 3 distinct edges based on
// nearest neighbour heuristic (and take the first improving perturbative neighbour)
inline void three_opt(vi& tour, const mi32& dist, const mi16& neighbour, uint_fast16_t* position) {
    uint_fast16_t n = tour.size();

    uint_fast16_t u1, u2, u3, u4, u5, u6;
    uint_fast16_t u1Idx, u2Idx, u3Idx, u4Idx, u5Idx, u6Idx;
    bool improvingTSP = true; // Exit when no improving neighbours

    int d0, d1, d2;

    /*
     * Exit SLS if no improving TSP (neighbours), i.e, we can't optimise the TSP any
     * further
     */
    while (improvingTSP) {
        improvingTSP = false;

        for (u1Idx = 0; u1Idx < n; ++u1Idx) {
            u1 = tour[u1Idx];
            u2Idx = (u1Idx + 1) % n;
            u2 = tour[u2Idx];

            for (uint_fast16_t k1 = 0; k1 < neighbour.getNumCols(); ++k1) {
                u3 = neighbour[u2][k1];
                u3Idx = position[u3];

                d0 = dist[u2][u3] - dist[u1][u2];

                // All subsequent neighbours of guaranteed to have longer distances
                if (d0 >= 0) break;

                if (u3 == u1) continue;

                u4Idx = (u3Idx + 1) % n;
                u4 = tour[u4Idx];

                if (u4Idx != u1Idx) {
                    for (uint_fast16_t k2 = 0; k2 < neighbour.getNumCols(); ++k2) {
                        u5 = neighbour[u4][k2];
                        u5Idx = position[u5];

                        if (u5 == u3 || u5 == u1 || u5 == u2) continue;

                        if (!((u5Idx < u3Idx && u3Idx < u2Idx) ||
                            (u3Idx < u2Idx && u2Idx < u5Idx) ||
                            (u2Idx < u5Idx && u5Idx < u3Idx))) continue;

                        d1 = d0 + dist[u4][u5] - dist[u3][u4];

                        // All subsequent neighbours of guaranteed to have longer distances
                        if (d1 >= 0) break;

                        u6Idx = (u5Idx + 1) % n;
                        u6 = tour[u6Idx];

                        d2 = d1 + dist[u1][u6] - dist[u5][u6];

                        if (d2 < 0) {
                            improvingTSP = true;

                            swap_adj_seg(tour, position, u4Idx, u1Idx, u2Idx, u5Idx);

                            goto nextU1U2;
                        }
                    }
                }

                u4Idx = ((u3Idx - 1) + n) % n;
                u4 = tour[u4Idx];

                if (u4Idx != u1Idx) {
                    for (uint_fast16_t k2 = 0; k2 < neighbour.getNumCols(); ++k2) {
                        u5 = neighbour[u4][k2];
                        u5Idx = position[u5];

                        if (u5 == u3 || u5 == u1 || u5 == u2) continue;

                        if (!((u5Idx < u3Idx && u3Idx < u2Idx) ||
                            (u3Idx < u2Idx && u2Idx < u5Idx) ||
                            (u2Idx < u5Idx && u5Idx < u3Idx))) continue;

                        d1 = d0 + dist[u4][u5] - dist[u3][u4];

                        // All subsequent neighbours of guaranteed to have longer distances
                        if (d1 >= 0) break;

                        u6Idx = (u5Idx + 1) % n;
                        u6 = tour[u6Idx];

                        d2 = d1 + dist[u1][u6] - dist[u5][u6];

                        if (d2 < 0) {
                            improvingTSP = true;

                            reverse_tour(tour, u6Idx, u4Idx, position);
                            reverse_tour(tour, u3Idx, u1Idx, position);

                            goto nextU1U2;
                        }
                    }
                }
            }
            nextU1U2: continue;
        }
    }
}

// Does local search over the problem until the time limit is reached
template<typename T>
vi local_search(mi32& dist, const chrono::time_point<T>& deadline) {
    uint_fast16_t n = dist.getNumRows();

    const mi16 neighbour = create_neighbour_matrix(dist, NEIGH_LIMIT);

    /*
     * Start of with the greedy solution (in problem specs), then improve from there via
     * SLS
     */
    vi tour = naive_algo(dist);

    // Compute cost for this tour, takes O(n)
    ll cost = compute_cost(tour, dist);

    // Perform 2-opt and 3 -top on tour
    vi bestTour = tour;
    ll bestCost = cost;

    //cout << "bestCost = " << bestCost << "\n";

    // Find finding our first local maxima, hope we get a pretty good one
    uint_fast16_t* position = new uint_fast16_t[n];

    uniform_int_distribution<size_t> randomOffset(1, 100); // [a, b] Inclusive

    chrono::milliseconds threeOptBuffer(THREE_OPT_BUFFER_TIME);

    // Diversification
    while (now() + threeOptBuffer < deadline) {
        size_t A = randomOffset(gen);
        if (A <= SHUFFLE_PROBABILITY) {
            shuffle_tour(tour);
        } else {
            tour = double_bridge(tour);
        }

        // Next, perform subsidiary local search
        for (uint_fast16_t i = 0; i < n; ++i) {
            position[tour[i]] = i;
        }

        // Intensification
        two_opt(tour, dist, neighbour, position);
        three_opt(tour, dist, neighbour, position);
        cost = compute_cost(tour, dist);

        // Acceptance criteria
        if (cost <= bestCost) {
            bestCost = cost;
            bestTour = tour;
        } else {
            tour = bestTour;
        }
    }

    return bestTour;
}

int main() {
    // Read TSP input into dist matrix
    mi32 dist = create_dist_matrix(cin);

    // Solve it
    vi tour = local_search(dist, now() + chrono::milliseconds(EXECUTION_DURATION));

    // Print the best tour
    print_tour(tour);
}
