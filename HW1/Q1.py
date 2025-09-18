from random import random

def estimate_pi(n_samples: int) -> float:
    """Estimate the value of π using the Monte Carlo method
    throw random points in the unit square and count how many fall withi the quarter circle of radius 1 
    and multiply the ratio by 4"""
    hits = 0
    for _ in range(n_samples):
        x = random()
        y = random()
        if x*x + y*y <= 1.0:
            hits += 1
    return 4.0 * hits / n_samples

# example:
if __name__ == "__main__":
    print(estimate_pi(1_000_000))