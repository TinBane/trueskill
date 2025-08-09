#!/usr/bin/env python
import argparse
import math
import random
import time
from typing import List, Tuple

from trueskill import TrueSkill, Rating, rate_1vs1
from trueskill.backends import available_backends


def _phi(x: float) -> float:
    # Standard normal CDF using error function
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _sample_match(players: List[Rating], team_size: int, rng: random.Random) -> Tuple[List[Rating], List[Rating], List[int], List[int]]:
    # sample 2 disjoint teams of size team_size
    indices = list(range(len(players)))
    rng.shuffle(indices)
    team_a_idx = indices[:team_size]
    team_b_idx = indices[team_size:2 * team_size]
    team_a = [players[i] for i in team_a_idx]
    team_b = [players[i] for i in team_b_idx]
    return team_a, team_b, team_a_idx, team_b_idx


def _team_mu(team: List[Rating]) -> float:
    return sum(r.mu for r in team)


def _team_perf_variance(team: List[Rating], beta: float) -> float:
    # Per TrueSkill, team performance variance: sum over players of (sigma^2 + beta^2)
    return sum((r.sigma ** 2) + (beta ** 2) for r in team)


def _decide_outcome(env: TrueSkill, team_a: List[Rating], team_b: List[Rating], draw_probability: float, rng: random.Random) -> Tuple[bool, int]:
    # Compute probability Team A beats Team B under current beliefs
    mu_diff = _team_mu(team_a) - _team_mu(team_b)
    c2 = _team_perf_variance(team_a, env.beta) + _team_perf_variance(team_b, env.beta)
    c = math.sqrt(c2)
    p_a_wins = _phi(mu_diff / c)
    # Draw branch first
    if rng.random() < draw_probability:
        return True, 0  # draw
    # Otherwise, decide winner by p_a_wins
    winner = 0 if rng.random() < p_a_wins else 1
    return False, winner


def benchmark(
    num_players: int,
    num_matches: int,
    team_size: int,
    draw_probability: float,
    inplace: bool,
    variant_1v1: bool,
    seed: int,
    backend: str,
    mu: float,
    sigma: float,
    beta: float,
    tau: float,
) -> None:
    rng = random.Random(seed)
    # Configure backend
    backend_choice = None if backend == 'python' else backend
    try:
        env = TrueSkill(
            mu=mu,
            sigma=sigma,
            beta=beta,
            tau=tau,
            draw_probability=draw_probability,
            backend=backend_choice,
        )
    except Exception as e:
        raise SystemExit(f"Requested backend '{backend}' is not available: {e}")

    # Initialize player ratings with identifiers for traceability
    players = [Rating(mu, sigma, identifier=f"P{i}") for i in range(num_players)]

    # Warm-up a few operations to avoid one-time overhead in timings
    for _ in range(5):
        a, b, a_idx, b_idx = _sample_match(players, team_size, rng)
        drawn, winner = _decide_outcome(env, a, b, draw_probability, rng)
        if team_size == 1 and variant_1v1:
            if drawn:
                na, nb = rate_1vs1(a[0], b[0], drawn=True, env=env, inplace=inplace)
                if not inplace:
                    players[a_idx[0]] = na
                    players[b_idx[0]] = nb
            else:
                if winner == 0:
                    na, nb = rate_1vs1(a[0], b[0], env=env, inplace=inplace)
                    if not inplace:
                        players[a_idx[0]] = na
                        players[b_idx[0]] = nb
                else:
                    nb, na = rate_1vs1(b[0], a[0], env=env, inplace=inplace)
                    if not inplace:
                        players[a_idx[0]] = na
                        players[b_idx[0]] = nb
        else:
            if drawn:
                teams = env.rate([tuple(a), tuple(b)], ranks=[0, 0], inplace=inplace)
            else:
                teams = env.rate([tuple(a), tuple(b)], ranks=[0, 1] if winner == 0 else [1, 0], inplace=inplace)
            if not inplace:
                # Map returned ratings back to original indices
                for j, r in enumerate(teams[0]):
                    players[a_idx[j]] = r
                for j, r in enumerate(teams[1]):
                    players[b_idx[j]] = r

    start = time.perf_counter()
    for _ in range(num_matches):
        team_a, team_b, a_idx, b_idx = _sample_match(players, team_size, rng)
        drawn, winner = _decide_outcome(env, team_a, team_b, draw_probability, rng)
        if team_size == 1 and variant_1v1:
            if drawn:
                ra, rb = rate_1vs1(team_a[0], team_b[0], drawn=True, env=env, inplace=inplace)
                if not inplace:
                    players[a_idx[0]] = ra
                    players[b_idx[0]] = rb
            else:
                if winner == 0:
                    ra, rb = rate_1vs1(team_a[0], team_b[0], env=env, inplace=inplace)
                    if not inplace:
                        players[a_idx[0]] = ra
                        players[b_idx[0]] = rb
                else:
                    rb, ra = rate_1vs1(team_b[0], team_a[0], env=env, inplace=inplace)
                    if not inplace:
                        players[a_idx[0]] = ra
                        players[b_idx[0]] = rb
        else:
            if drawn:
                teams = env.rate([tuple(team_a), tuple(team_b)], ranks=[0, 0], inplace=inplace)
            else:
                teams = env.rate([tuple(team_a), tuple(team_b)], ranks=[0, 1] if winner == 0 else [1, 0], inplace=inplace)
            if not inplace:
                for j, r in enumerate(teams[0]):
                    players[a_idx[j]] = r
                for j, r in enumerate(teams[1]):
                    players[b_idx[j]] = r
    elapsed = time.perf_counter() - start

    matches_per_sec = num_matches / elapsed if elapsed > 0 else float('inf')
    print('Benchmark complete:')
    print(f'  players           : {num_players}')
    print(f'  team size         : {team_size}')
    print(f'  matches           : {num_matches}')
    print(f'  inplace updates   : {inplace}')
    print(f'  1v1 fast-path     : {variant_1v1}')
    print(f'  backend           : {backend}')
    print(f'  draw probability  : {draw_probability}')
    print(f'  env (mu,sigma,beta,tau) = ({mu},{sigma},{beta},{tau})')
    print(f'  elapsed (s)       : {elapsed:.4f}')
    print(f'  throughput (matches/s): {matches_per_sec:,.0f}')


def main():
    parser = argparse.ArgumentParser(description='TrueSkill benchmark: simulate and rate many matches.')
    parser.add_argument('--players', type=int, default=1000, help='Number of players in the pool')
    parser.add_argument('--matches', type=int, default=100000, help='Number of matches to simulate')
    parser.add_argument('--team-size', type=int, default=1, help='Number of players per team (1 for 1v1, 2 for 2v2, etc.)')
    parser.add_argument('--draw-prob', type=float, default=0.10, help='Probability of draw used in simulation and environment')
    parser.add_argument('--inplace', action='store_true', help='Update ratings in place')
    parser.add_argument('--no-1v1-fastpath', action='store_true', help='Disable rate_1vs1 fast path for 1v1 (use env.rate instead)')
    parser.add_argument('--seed', type=int, default=1729, help='Random seed')
    parser.add_argument('--backend', choices=['python', 'fast', 'scipy', 'mpmath'], default='python', help='Math backend to use')
    parser.add_argument('--mu', type=float, default=25.0)
    parser.add_argument('--sigma', type=float, default=25.0/3.0)
    parser.add_argument('--beta', type=float, default=(25.0/3.0)/2.0)
    parser.add_argument('--tau', type=float, default=(25.0/3.0)/100.0)

    args = parser.parse_args()
    # Print available backends to aid debugging
    print('available_backends:', available_backends())
    benchmark(
        num_players=args.players,
        num_matches=args.matches,
        team_size=args.team_size,
        draw_probability=args.draw_prob,
        inplace=args.inplace,
        variant_1v1=(not args.no_1v1_fastpath),
        seed=args.seed,
        backend=args.backend,
        mu=args.mu,
        sigma=args.sigma,
        beta=args.beta,
        tau=args.tau,
    )


if __name__ == '__main__':
    main()


