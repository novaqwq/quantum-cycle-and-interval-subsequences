"""Exact checks for periodically observed nearest-neighbour quantum cycles.

Run with Python 3.9+; no third-party packages. The order-60 certificate is
the finite step in the proof. The subsequent range check is independent of
that reduction and computes every worst-case multiplicity directly.
"""

from argparse import ArgumentParser
from collections import defaultdict
from functools import lru_cache, reduce
from itertools import combinations
from math import comb, gcd


def exact_quotient(dividend, divisor):
    """Divide by a monic integer polynomial, coefficients in ascending order."""
    remainder = list(dividend)
    quotient = [0] * (len(dividend) - len(divisor) + 1)
    for shift in reversed(range(len(quotient))):
        coefficient = remainder[shift + len(divisor) - 1]
        quotient[shift] = coefficient
        if coefficient:
            for i, value in enumerate(divisor):
                remainder[shift + i] -= coefficient * value
    if any(remainder):
        raise ArithmeticError("Polynomial division was not exact")
    return tuple(quotient)


@lru_cache(maxsize=None)
def cyclotomic(n):
    polynomial = (-1,) + (0,) * (n - 1) + (1,)
    for divisor in range(1, n):
        if n % divisor == 0:
            polynomial = exact_quotient(polynomial, cyclotomic(divisor))
    return polynomial


def energies(n):
    """Represent z^k + z^(-k) in the power basis of Q[z]/Phi_n(z)."""
    modulus = cyclotomic(n)
    degree = len(modulus) - 1
    one = (1,) + (0,) * (degree - 1)
    powers = [one]
    for _ in range(n):
        previous = powers[-1]
        shifted = (0,) + previous[:-1]
        powers.append(tuple(a - previous[-1] * b for a, b in zip(shifted, modulus)))
    if powers[-1] != one:
        raise ArithmeticError("Primitive-root power table failed the period check")
    levels = [tuple(a + b for a, b in zip(powers[k], powers[n - k]))
              for k in range(n // 2 + 1)]
    if len(set(levels)) != len(levels):
        raise ArithmeticError("Repeated energy in the chosen half spectrum")
    return levels


def direction(x, y):
    difference = tuple(b - a for a, b in zip(x, y))
    common = reduce(gcd, difference, 0)
    first = next(value for value in difference if value)
    signed_divisor = common if first > 0 else -common
    return tuple(value // signed_divisor for value in difference)


def order_sixty_certificate():
    levels = energies(60)
    rational_indices = [i for i, x in enumerate(levels) if not any(x[1:])]
    zero = (0,) * len(levels[0])
    zero_index = levels.index(zero)
    expected_integers = set(combinations(rational_indices, 3))
    expected_irrational = {
        tuple(sorted((i, zero_index, levels.index(tuple(-v for v in x)))))
        for i, x in enumerate(levels) if i not in rational_indices
    }
    observed = {(i, j, k) for i, j, k in combinations(range(len(levels)), 3)
                if direction(levels[i], levels[j]) == direction(levels[i], levels[k])}
    if observed != expected_integers | expected_irrational:
        raise AssertionError("Unexpected or missing rational-affine triple at order 60")
    print(f"C_60: {comb(len(levels), 3)} triples; {len(expected_integers)} integer triples, "
          f"{len(expected_irrational)} irrational symmetric triples, 0 others.")


def maximum_phase_multiplicity(n):
    levels = energies(n)
    weights = [1 if 2 * k % n == 0 else 2 for k in range(len(levels))]
    maximum = max(weights)
    for i, anchor in enumerate(levels):
        other_weights = defaultdict(int)
        for j in range(i + 1, len(levels)):
            ray = direction(anchor, levels[j])
            other_weights[ray] += weights[j]
        maximum = max(maximum, weights[i] + max(other_weights.values(), default=0))
    return maximum


def proposed_count(n):
    if n <= 4:
        return n
    if n % 12 == 0:
        return 8
    if n % 4 == 0 or n % 6 == 0:
        return 6
    return 4


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--max-n", type=int, default=120)
    args = parser.parse_args()
    if args.max_n < 3:
        parser.error("--max-n must be at least 3")
    order_sixty_certificate()
    for n in range(3, args.max_n + 1):
        observed = maximum_phase_multiplicity(n)
        if observed != proposed_count(n):
            raise AssertionError(f"n={n}: observed {observed}, expected {proposed_count(n)}")
    print(f"Exact maximum phase multiplicities agree for every n from 3 to {args.max_n}.")
