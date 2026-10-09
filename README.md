# gnssdl

Tools for daily GNSS position time series from the
[Nevada Geodetic Laboratory](https://geodesy.unr.edu) (NGL), and a benchmark
for filtering them.

The project measures what network filtering of daily GNSS series removes.
Filters that subtract noise shared across a network also remove some real
deformation signal, and can create signal that was never there. The
benchmark predicts each station from the other stations (leave-one-out) on
~1,250 California stations and scores every filter on three things:
- noise removed;
- planted transients kept;
- spurious signal created.

The method and motivation are in [`paper/`](paper/), and the design of each
model and filter is in [`docs/`](docs/).

## Install

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
uv sync
uv run gnssdl --help
uv run pytest            # tests
```

## Overview

| Command | Does |
|---|---|
| `gnssdl meta` | download NGL's station table and step catalogue |
| `gnssdl near` | list stations near a point, in a box, or covering a date |
| `gnssdl fetch` | download daily series (`.tenv3`) |
| `gnssdl departure` | plot departures from a reference trajectory |
| `gnssdl build` | build the benchmark data cube |
| `gnssdl bench masks` | generate the fixed evaluation masks |
| `gnssdl bench run` | fit and score a model on the masks |
| `gnssdl bench signal` | planted-signal, Ridgecrest and noise-removed tests |
| `gnssdl bench report` | print the scoreboard |
| `gnssdl bench compare` | paired bootstrap comparison of two models |
| `gnssdl bench kernel` | plot a trained M3k kernel |

Global options go before the subcommand:

- `--data-dir DIR`: where data is stored (default `data/`).
- `--refresh`: re-download even if cached.

Downloads are cached: rerunning a command fetches only what is missing.

## Data

### `gnssdl meta`

Downloads `DataHoldings.txt` (station table) and `steps.txt` (equipment
changes and earthquakes that may offset each station) into `data/meta/`.
Every other command does this on first use.

### `gnssdl near`

Lists stations by location and data span.

| Option | Meaning |
|---|---|
| `--lat`, `--lon`, `-r KM` | within `KM` of a point (default 200 km) |
| `--bbox S N W E` | inside a lat/lon box instead |
| `--min-days N` | at least N daily solutions |
| `--around DATE` | record covers `--before` days before to `--after` days after DATE (defaults 365 and 90) |

```bash
uv run gnssdl near --lat 35.77 --lon -117.60 -r 40
uv run gnssdl near --lat 37.23 --lon 37.01 -r 150 --around 2023-02-06
uv run gnssdl near --bbox 32.4 42.1 -124.6 -114.0 --min-days 1095
```

`--around` uses each station's first and last solution date, so a station
can pass and still have gaps inside the window.

### `gnssdl fetch`

Downloads IGS20 daily `.tenv3` series to `data/tenv3/`, either for named
stations or for the same selections as `near`.

| Option | Meaning |
|---|---|
| `STATION ...` | 4-character station ids |
| selection options | as in `near` |
| `--manifest FILE` | write the selection with coordinates, span and download status as CSV |
| `-j N` | parallel downloads (default 4; NGL is a university server) |

```bash
uv run gnssdl fetch P595 CCCC GOLD
uv run gnssdl fetch --bbox 32.4 42.1 -124.6 -114.0 --min-days 1095 \
    --manifest data/manifests/california.csv     # the benchmark set, ~1.6 GB
```

Use the manifest, not the folder listing, as the station list of a
selection. NGL updates its solutions and posts new days about 10 days
late; record the download date with any result.

### `gnssdl departure`

Fits constant + rate + annual + semi-annual + steps per station and
component over a reference window, and plots the data minus that model in
mm, including outside the window.

| Option | Meaning |
|---|---|
| `--fit START END` | reference window (required) |
| `--event DATE:LABEL` | vertical marker (repeatable) |
| `--xlim START END` | plotted range |
| `--title`, `-o FILE` | title and output (default `figures/departure.png`) |

```bash
uv run gnssdl departure P595 P594 CCCC GOLD \
    --fit 2011-01-01 2016-12-31 \
    --event "2019-07-04:Ridgecrest Mw6.4" --event "2019-07-06:Ridgecrest Mw7.1" \
    --xlim 2018-07-01 2021-06-30 -o figures/ridgecrest.png
```

How steps are handled:
- Steps inside the window are estimated.
- Equipment changes after the window are removed with a local ±30-day jump
  estimate and marked ▲.
- Earthquake offsets after the window are left in.

## Benchmark

The full specification is
[`docs/00-benchmark-contract.md`](docs/00-benchmark-contract.md). A complete
run, from fetched data to a scoreboard:

```bash
uv run gnssdl build                      # data/cube/california.npz (+ .json summary)
uv run gnssdl bench masks                # data/cube/masks.npz
uv run gnssdl bench run m1               # fit and score at every radius
uv run gnssdl bench signal m1 --centres 20
uv run gnssdl bench report
```

### `gnssdl build`

Builds the data cube from the stations in `--manifest` (default
`data/manifests/california.csv`) into `-o` (default
`data/cube/california.npz`). It contains:
- residuals after a trajectory fit on 2008–2019;
- availability, splits and held-out stations;
- the 256 nearest neighbours of each station beyond each exclusion radius.

Station QC, twin removal and the dropped stations are reported in the
`.json` summary next to the cube.

| Option | Meaning |
|---|---|
| `--end DATE` | last day of the cube (default: latest epoch) |
| `--freeze-date DATE` | days after DATE become the prospective test period |

Splits: train 2008–2019, validation 2020–2021, test 2022 to the freeze
date, prospective after it. Decisions use validation only.

### `gnssdl bench masks`

Writes the fixed evaluation masks (scattered days, 30-day blocks, held-out
stations, the Ridgecrest year) to `data/cube/masks.npz`. Run once per cube.

### `gnssdl bench run MODEL`

Fits a model (tuning on validation where it has a hyper-parameter),
checks that it never reads hidden cells or stations within the exclusion
radius, and scores it on the masks. Results go to
`data/results/MODEL.csv` and `.json`.

`--radius KM` (repeatable) limits the exclusion radii; the default is every
radius of the cube.

| Model | What it is |
|---|---|
| `m0` | predict zero (no filtering) |
| `m1` | distance-weighted mean of the 16 nearest stations beyond R |
| `m1k64`, `m1k256` | M1 with 64 or 256 neighbours |
| `m2` | robust (weighted median) neighbour statistic |
| `d1` | mean of all stations at least 400 km away |
| `fs` | the same far stack at R = 0, 25, 50, 100, 200, 400, 600 km |
| `m3k`, `m3k256` | learned kernel: a small network sets each neighbour's weight from its distance, direction and noise level (16 or 256 neighbours) |

Learned models train on the GPU when one is available and save their
weights and training history to `data/models/MODEL/`; follow training with
`tail -F data/models/MODEL/train.log`.

### `gnssdl bench signal MODEL`

Plants synthetic transients in real data and measures how much survives
filtering (ρ, 1 = kept) and how much spurious signal appears at stations
where nothing was planted. It also reports Ridgecrest postseismic
retention and noise removed. Results go to
`data/results/signal/PERIOD/MODEL_*.csv`.

| Option | Meaning |
|---|---|
| `--period` | `validation` (default, for decisions), `test` or `prospective` |
| `--centres N` | injections per grid cell (default 50; 20 is enough for design runs) |
| `--radius KM` | as in `run` |
| `--noise-only` | only the noise-removed measure (fast) |

Besides the `run` models, it accepts the reference filters. These read the
target station's own data, so they are never scored on the masks:

| Filter | What it is |
|---|---|
| `m2self` | M2 with the target included |
| `c1` | mean of the whole network, target included |
| `c3` | sub-regional probabilistic PCA, swept over 1, 4, 8 regions × 1–3 components |

### `gnssdl bench report`

Prints the scoreboard: median normalised error per model and radius, then
signal kept, spurious signal and noise removed.

| Option | Meaning |
|---|---|
| `--split` | `validation` (default), `test` or `prospective` |
| `--part` | `fast`, `slow`, `total` or `all` (default) |

### `gnssdl bench compare A B`

Paired bootstrap comparison of two models: 95% intervals for the difference
in error (resampling 30-day blocks and stations) and in signal kept
(resampling injections).

| Option | Meaning |
|---|---|
| `--radius KM` | radius of A (and of B unless `--radius-b` is given) |
| `--radius-b KM` | radius of B |
| `--period` | as in `signal` |

```bash
uv run gnssdl bench compare m1 d1 --radius 0
```

### `gnssdl bench kernel MODEL`

Plots a trained M3k kernel (`m3k` or `m3k256`) into `-o` (default
`figures/m3k/`):
- weight against distance, compared with M1's Gaussian;
- which directions get more weight;
- weight against the neighbour's noise level;
- the kernel at several exclusion radii.

Curves cover only the distances of the neighbours the model was trained on.

| Option | Meaning |
|---|---|
| `--radius KM` | exclusion radius of the first three figures (default 0) |
| `--seed N` | which training seed to plot (default 0) |

## Python API

```python
from gnssdl import ngl, model

series = ngl.read_tenv3("data/tenv3/P595.tenv3")   # DataFrame by date: e, n, u (m), sigmas, ...
steps = ngl.read_steps("data/meta/steps.txt")
stations = ngl.read_holdings("data/meta/DataHoldings.txt")

mine = steps[steps.sta == "P595"]
resid_mm, vel_mm_yr = model.departure(
    series, "2011-01-01", "2016-12-31",
    step_dates=sorted(set(mine.date)),
    equipment_dates=sorted(set(mine[mine.kind == "equipment"].date)),
)
```

## Layout

```
src/gnssdl/
  ngl.py        NGL URLs, cached downloads, parsers, station selection
  model.py      trajectory model and departures
  plots.py      departure plots
  dataset.py    data cube (gnssdl build)
  cli.py        the gnssdl command
  bench/        masks, models, audit filters, scoring, signal tests, statistics
tests/          unit tests on synthetic data
docs/           benchmark contract and per-model design docs
paper/          research proposal (LaTeX; `make` builds proposal.pdf)
data/           downloaded data and results, not tracked
figures/        generated plots, not tracked
```

## Data and attribution

All GNSS time series are from the Nevada Geodetic Laboratory, which this
repository does not redistribute. Cite NGL as they request, for example:

> Blewitt, G., Hammond, W. C., & Kreemer, C. (2018). Harnessing the GPS data
> explosion for interdisciplinary science. *Eos*, 99.
> https://doi.org/10.1029/2018EO104623

## License

Code: MIT ([`LICENSE`](LICENSE)). Research proposal in `paper/`: CC BY 4.0
([`paper/LICENSE`](paper/LICENSE)).

## Author

H. Oğuz Bolat, Department of Geomatics Engineering, Yıldız Technical
University (oguz.bolat@std.yildiz.edu.tr)
