# Auditing a filter of your own

This page is for someone who has a common-mode filter and wants its report card: how much noise it
removes, how much planted signal it keeps, how much signal it invents, with intervals, against the
reference filter. It assumes a built cube (`gnssdl build`, see the README) and nothing else.
`examples/median_filter.py` is the complete example the snippets below are taken from.

## 1. Write the filter

A filter is a subclass of `gnssdl.Filter` with one method:

```python
import numpy as np
import gnssdl
from gnssdl.dataset import distance_azimuth

class FarMedian(gnssdl.Filter):
    name = "farmedian"
    never_reads_target = True

    def predict(self, cube, hide):
        values = np.where(cube.avail[..., None], cube.r, np.nan)
        distance, _ = distance_azimuth(cube.lat, cube.lon)
        out = np.zeros(values.shape, dtype=np.float32)
        for i in range(len(cube.sta)):
            far = distance[i] > self.radius_km
            far[i] = False
            if far.any():
                with np.errstate(all="ignore"):
                    out[i] = np.nan_to_num(np.nanmedian(values[far], axis=0))
        return out
```

`predict` returns the filter's estimate of the common mode, an array of stations × days × 3
(east, north, up) in millimetres. The cleaned series the benchmark scores is `cube.r` minus that.

What `predict` receives:

| | |
|---|---|
| `cube.r` | stations × days × 3, mm; NaN where there is no solution |
| `cube.avail` | stations × days, True where a value may be read. Read `cube.r` only where this is True |
| `cube.sigma` | formal uncertainties, same shape as `cube.r` |
| `cube.sta`, `cube.lat`, `cube.lon`, `cube.days` | station names and coordinates, and the days |
| `cube.nbr_idx`, `cube.nbr_dist`, `cube.nbr_az` | for each exclusion radius of the cube (`cube.radius_index(R)`), each station's nearest stations beyond that radius, nearest first, with distance and azimuth; -1 pads |
| `hide` | stations × days, the cells the harness has hidden; they are already unavailable in `cube` |
| `self.radius_km` | the exclusion radius of this run: stations nearer than this to the target are not to be used |

`name` is the filter's name on the command line, its results directory and the `model` column of
every results file. Use lower-case letters, digits and underscores, and do not change it once
results exist.

## 2. Say whether it reads the target station

The noise and signal measures mean different things for a filter that cleans a station using that
station's own values and one that does not, so a filter declares which it is:

- `never_reads_target = True`: the prediction for a station never uses that station's own values
  (and nothing nearer than `radius_km`). The harness then cleans every station in one pass. This
  claim is checked before anything is scored: the target and the stations inside the radius are
  overwritten with a huge value, and the prediction must not move. A filter that fails is stopped
  with `failed leak checks`.
- `reference_filter = True`: the filter reads the target on purpose, as a network stack or a PCA
  over all stations does. It is scored by the signal tests only, and its noise-removed figure
  includes the target's own noise, which the card keeps apart from the leave-one-out filters.

A filter that declares neither is refused.

## 3. Settings fitted to the data (optional)

A filter with nothing to fit needs only `predict`. One that learns from the training years
implements `fit(self, cube, val_hide=None, scorer=None)`, called once before `predict`; training
days are `cube.split_day == 0`, and cells with `cube.exclude` are barred from training. To let the
harness tune a setting on the validation masks the same way for every filter:

- list it in `hyperparams = ("length_km",)` and accept it in `__init__` (default `None`);
- in `fit`, when it is `None`, predict on the cube with `val_hide` hidden
  (`gnssdl.bench.base.apply_hide(cube, val_hide)`) for each candidate value and keep the one with
  the lowest `scorer(prediction)`;
- return it from `config()` (`{**super().config(), "length_km": self.length_km}`), so it is saved
  with the results and passed back to `__init__` when the filter is loaded again.

`gnssdl/bench/m1_stack.py` is a short built-in filter that does exactly this.

## 4. Run the audit

```python
net = gnssdl.Network.load("pnw")                       # or a cube path
card = gnssdl.audit(FarMedian, net, centres=5, log=print)
print(card)
```

`audit` makes the evaluation masks if the network has none, fits the filter and runs the leak
checks, runs the signal tests for it, runs them for the reference (the far stack at R = 0) if that
has no results yet, and returns the card. Results are saved under the network's results root
(`data/runs`), and a second call runs only what is missing.

| Argument | Meaning |
|---|---|
| `radii` | exclusion radii in km; default `(0.0,)`, where the criterion's reference sits; `"all"` for every radius |
| `centres` | planted transients per grid cell. The benchmark uses 50; a small number gives a quick, noisier card |
| `measures` | which signal tests: `"transients"` (with noise removed), `"seasonal"`, `"offsets"`, `"episodes"` |
| `rerun` | recompute what is already saved |
| `seed`, `train` | for trained filters; `audit` never trains by itself unless `train=True` |
| `log` | a function that receives progress lines, e.g. `print` |

The validation period is the only one available before the method is frozen; the test period is
refused (contract §1.4).

## 5. Read the card

`print(card)` shows each measure with a 95% bootstrap interval for the filter and the reference,
then the verdict. `card.table()` gives one row per filter and radius, `card.measures` every
interval (all transient footprints), `card.verdict()` the verdict alone.

| Measure | Meaning | Better is |
|---|---|---|
| `noise_removed` | fraction of the fast (under about two months) scatter the filter removes, median over stations | higher |
| `error` | leave-one-out error of the prediction, fast part; absent for filters that read the target | lower |
| `kept` | fraction of a planted transient left in the cleaned series, median over transients; shown for the 100-km footprint | nearer 1 |
| `spurious` | largest false movement the filter creates at stations outside a planted transient, as a fraction of its amplitude | nearer 0 |
| `seasonal` | fraction of each station's annual and semi-annual cycle left in the cleaned series | reported only |
| offsets, episodes | fraction of real earthquake offsets and of slow-slip episodes kept, where the network has them | reported only |

The verdict applies the success criterion (contract §1.6) against the far stack at R = 0:

- `better`: noise removed is higher by at least 0.01 with an interval above zero, and 100-km kept
  is not lower by more than 0.05, and 100-km spurious is not higher by more than 0.05;
- `noise gain bought with signal`: noise removed is higher in that sense, but one of the two signal
  conditions fails;
- `not better`: the gain in noise removed is below 0.01, or the interval of the difference includes zero.

## 6. Use it from the command line

Registering the filter makes its name valid for every `gnssdl bench` command and for queue
manifests:

```python
from gnssdl.bench import registry
registry.register(FarMedian, title="Far median", radii=(0.0, 50.0, 100.0))
```

To have the `gnssdl` command find it without an import, the package that ships the filter declares
an entry point in its `pyproject.toml`, pointing at a function that calls `register`:

```toml
[project.entry-points."gnssdl.filters"]
farmedian = "mypackage.filters:register_all"
```
