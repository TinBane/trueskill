# TrueSkill, the video game rating system
***
[![Build Status](https://img.shields.io/travis/sublee/trueskill.svg)
](https://travis-ci.org/sublee/trueskill)
[![Coverage Status](https://img.shields.io/coveralls/sublee/trueskill.svg)
](https://coveralls.io/r/sublee/trueskill)

See [the documentation](http://trueskill.org/).

* by [Heungsub Lee](http://subl.ee/)
* improvements by Adrian Beale

***
## TrueSkill

An implementation of the TrueSkill algorithm for Python.  TrueSkill is a rating
system among game players and it is used on Xbox Live to rank and match
players.

```python
from trueskill import Rating, quality_1vs1, rate_1vs1

alice, bob = Rating(25), Rating(30)  # assign Alice and Bob's ratings
if quality_1vs1(alice, bob) < 0.50:
    print('This match seems to be not so fair')
alice, bob = rate_1vs1(alice, bob)  # update the ratings after the match
```

### Optional fast backend (Cython)

You can speed up TrueSkill by using a Cython-powered math backend for the
normal CDF/PDF/PPF primitives.

- Install with fast extra (builds the extension during install):
  ```
  pip install trueskill[fast]
  ```
  Or inside Pipenv:
  ```
  pipenv install trueskill[fast]
  ```

- Alternatively, build locally after cloning:
  ```bash
  pipenv run pip install setuptools Cython
  pipenv run python setup.py build_ext --inplace
  ```

- Use the backend at runtime:
  ```python
  from trueskill import setup
  setup(backend='fast')  # falls back to pure Python if the extension is missing
  ```

If SciPy is available, you can also get a speedup via:
```python
from trueskill import setup
setup(backend='scipy')
```

### In-place updates and identifiers

`Rating` supports an optional identifier and in-place updates to avoid
re-associating objects in your application:

```python
from trueskill import Rating, rate_1vs1, TrueSkill, rate

alice = Rating(25, identifier='alice-uuid')
bob = Rating(25, identifier='bob-uuid')

# Update in place (original objects are modified and returned)
alice, bob = rate_1vs1(alice, bob, inplace=True)

# Many-vs-many in place
env = TrueSkill()
team_a = (Rating(25, identifier='A1'), Rating(25, identifier='A2'))
team_b = (Rating(25, identifier='B1'), Rating(25, identifier='B2'))
env.rate([team_a, team_b], ranks=[0, 1], inplace=True)
```

If `inplace=False` (default), functions return new `Rating` objects.

### Benchmarking

There is a simple benchmarking script to measure throughput of rating updates
with configurable parameters.

Examples:

```bash
# 1v1, in-place updates
python scripts/benchmark.py \
  --players 2000 --matches 200000 --team-size 1 --inplace

# 2v2, non-inplace
python scripts/benchmark.py \
  --players 2000 --matches 100000 --team-size 2

# Disable 1v1 fast path and set env parameters
python scripts/benchmark.py \
  --players 1000 --matches 50000 --team-size 1 --no-1v1-fastpath \
  --mu 25 --sigma 8.3333333333 --beta 4.1666666667 --tau 0.0833333333

Force a specific math backend and fail if unavailable using `--backend`:


# python-native (pure) backend
python scripts/benchmark.py --players 400 --matches 20000 --team-size 1 --inplace --backend python

# fast (Cython) backend
python scripts/benchmark.py --players 400 --matches 20000 --team-size 1 --inplace --backend fast

On a sample run (Apple Silicon, Python 3.12), 1v1 in-place, 400 players, 20k matches:


python  backend: ~5.6k matches/s
fast    backend: ~11.4k matches/s
```

The script exits with an error if the requested backend is not available. Fast Cython backend will have more impact on older versions of python (pre 3.11ish), python performance has somewhat caught up and we have typing hints to enable native python performance improvements.

Flags:
- `--players`, `--matches`, `--team-size`
- `--draw-prob` (simulation and environment)
- `--inplace` (in-place updates)
- `--no-1v1-fastpath` (use generic path even for 1v1)
- `--seed`, `--mu`, `--sigma`, `--beta`, `--tau`


***
## Links

Documentation
   http://trueskill.org/
GitHub:
   http://github.com/sublee/trueskill
Mailing list:
   trueskill@librelist.com
List archive:
   http://librelist.com/browser/trueskill
Continuous integration (Travis CI)
   https://travis-ci.org/sublee/trueskill

   .. image:: https://api.travis-ci.org/sublee/trueskill.png

####See Also:
- TrueSkill(TM) Ranking System by Microsoft
  <http://research.microsoft.com/en-us/projects/trueskill/>
- "Computing Your Skill" by Jeff Moser <http://bit.ly/moserware-trueskill>
- "The Math Behind TrueSkill" by Jeff Moser <http://bit.ly/trueskill-math>
- TrueSkill Calcurator by Microsoft
  <http://atom.research.microsoft.com/trueskill/rankcalculator.aspx>