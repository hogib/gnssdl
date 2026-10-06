# gnssdl

Tools and a planned deep-learning benchmark for daily GNSS position time
series, built on the open solutions of the
[Nevada Geodetic Laboratory](https://geodesy.unr.edu) (NGL).

The research question is how to separate noise shared across a GNSS network
from real deformation signal. Classical filters and recent deep-learning
denoisers both remove shared noise. The open question is how much real
transient signal they remove along with it, and whether a deep network does
better than linear spatial filtering once that is measured. The approach is a
self-supervised fill-in-the-blank task on the ~1,400-station California
network:
1. hide stations and days;
2. predict them from the neighbours;
3. score the models on prediction error and on whether synthetic transients
   planted in the data survive.

The full motivation and method are in [`paper/`](paper/). The design of each
model is in [`docs/`](docs/).

## Status

| Part | State |
|---|---|
| NGL data fetching (`gnssdl meta`, `near`, `fetch`) | Implemented, tested |
| Trajectory model and departure plots (`gnssdl departure`) | Implemented, tested |
| Research proposal (`paper/`) | Draft |
| Benchmark data cube, masks, scoring | Designed ([`docs/00-benchmark-contract.md`](docs/00-benchmark-contract.md)), not implemented |
| Models M0–M5 | Designed ([`docs/`](docs/)), not implemented |

## Install

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
uv sync
uv run gnssdl --help
```

## Usage

All data goes to `data/` (override with `--data-dir`, which goes before the
subcommand). Downloads are cached: rerunning a command fetches only what is
missing, and `--refresh` forces a re-download.

### Station metadata

```bash
uv run gnssdl meta
# 23812 stations, 142597 steps (24147 equipment, 118450 quake)
```

This downloads NGL's station table (`DataHoldings.txt`) and offset catalogue
(`steps.txt`) into `data/meta/`. Every other command does this automatically
on first use.

### Find stations

```bash
# within 40 km of a point
uv run gnssdl near --lat 35.77 --lon -117.60 -r 40

# within 150 km, keeping only stations whose record covers 365 days before
# to 90 days after a date
uv run gnssdl near --lat 37.23 --lon 37.01 -r 150 --around 2023-02-06

# inside a lat/lon box (S N W E), at least 3 years of data
uv run gnssdl near --bbox 32.4 42.1 -124.6 -114.0 --min-days 1095
```

`--around` uses each station's first and last solution date, so a station
can pass the filter and still have gaps inside the window.

### Download daily series

```bash
# named stations
uv run gnssdl fetch P595 CCCC GOLD

# the same selections as `near`
uv run gnssdl fetch --lat 37.23 --lon 37.01 -r 150 --around 2023-02-06

# the California set used by the benchmark (~1,380 stations, ~1.6 GB)
uv run gnssdl fetch --bbox 32.4 42.1 -124.6 -114.0 --min-days 1095 \
    --manifest data/manifests/california.csv
```

Series are IGS20 daily `.tenv3` files saved to `data/tenv3/`. `--manifest`
writes the selected stations with their coordinates, data span and download
status. Use the manifest, not the folder listing, as the station list for a
given selection. Downloads use 4 parallel connections by default (`-j`); NGL
is a university server.

### Plot departures from a reference trajectory

```bash
uv run gnssdl departure P595 P594 CCCC GOLD \
    --fit 2011-01-01 2016-12-31 \
    --event "2019-07-04:Ridgecrest Mw6.4" --event "2019-07-06:Ridgecrest Mw7.1" \
    --xlim 2018-07-01 2021-06-30 \
    -o figures/ridgecrest.png
```

For each station and component this fits

  constant + rate + annual + semi-annual + steps

over the `--fit` window. It then plots the data minus that model, in mm,
including outside the window. Anything the reference period didn't contain
shows up as a departure from zero, for example:
- a coseismic offset
- postseismic relaxation
- a change in rate
- a seasonal anomaly

Steps from NGL's catalogue inside the window are estimated. Equipment changes
after the window are removed with a local ±30-day jump estimate and marked ▲,
so an antenna swap cannot pass for a precursor. Earthquake offsets after the
window are deliberately left in.

## Python API

```python
from gnssdl import ngl, model

series = ngl.read_tenv3("data/tenv3/P595.tenv3")   # DataFrame indexed by date: e, n, u (m), sigmas, ...
steps = ngl.read_steps("data/meta/steps.txt")      # equipment and earthquake steps per station
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
  ngl.py      NGL URLs, cached downloads, parsers for DataHoldings / steps / tenv3, station selection
  model.py    trajectory model and departures
  plots.py    stations × components departure grids
  cli.py      the gnssdl command
tests/        parsers, selection, trajectory model (synthetic)
docs/         benchmark contract and per-model design docs (M0–M5)
paper/        research proposal (LaTeX); `make` builds proposal.pdf
data/         downloaded data, not tracked
figures/      generated plots, not tracked
```

## Tests

```bash
uv run pytest
```

## Building the proposal

```bash
cd paper && make
```

`make` uses tectonic if it's installed, otherwise latexmk, otherwise
pdflatex with bibtex.

## Data and attribution

All GNSS time series are from the Nevada Geodetic Laboratory. If you use data
fetched with this tool, cite NGL as they request on their website, for
example:

> Blewitt, G., Hammond, W. C., & Kreemer, C. (2018). Harnessing the GPS data
> explosion for interdisciplinary science. *Eos*, 99.
> https://doi.org/10.1029/2018EO104623

NGL updates and reprocesses its solutions. Results depend on the download
date, which is recorded in each file's modification time and should be
reported.

## License

- Code: MIT, see [`LICENSE`](LICENSE).
- Research proposal in [`paper/`](paper/): CC BY 4.0, see
  [`paper/LICENSE`](paper/LICENSE). The Makefile there is code and falls under
  MIT.

The license does not cover NGL data, which this repository does not
redistribute. See "Data and attribution" above.

## Author

H. Oğuz Bolat, Department of Geomatics Engineering, Yıldız Technical
University (oguz.bolat@std.yildiz.edu.tr)
