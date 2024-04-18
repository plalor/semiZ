# semiZ 

#### semiZ is a repository for estimating the atomic number of dual energy X-ray images using a semiempirical transparency model

## What do I need for this repository?

#### To use this repository, you will need to provide:
* `phi_H` -- an approximate model of the scanner's high energy beam spectrum.
* `phi_L` -- an approximate model of the scanner's low energy beam spectrum.
* `D` -- an approximate detector response function.
* `E` -- an energy grid on which `phi_H`, `phi_L`, and `D` are defined.

#### Furthermore, you will need a set of at least three experimental calibration measurements:
* `alpha_H_calib` -- measured log-transparencies using the high energy beam.
* `alpha_L_calib` -- measured log-transparencies using the low energy beam.
* `lmbda_calib` -- the area density of the calibration objects.
* `Z_calib` -- the atomic number of the calibration objects.

#### Then, for precomputing the forward model, you need to define a `{lambda, Z}` mesh:
* `lmbda_range` -- typically `np.linspace(0, 300, 301)`.
* `Z_range` -- typically `np.arange(1, 101)`.

#### Lastly, you will need a dual energy radiographic image:
* `im_H` -- log-transparency image using the high energy beam.
* `im_L` -- log-transparency image using the low energy beam.
* `labels` -- image segmentation map for identifying clusters of pixels which make up different objects. For example, by running `skimage.segmentation.felzenswalb` on the raw dual energy image.

## How do I use this repository?

#### Step zero: import the necessary functions

```python:
from semiZ import fitSemiempirical, calcLookupTables, Lookup, calcZ
```

#### Step one: perform the calibration, i.e. find the parameters `a`, `b`, and `c` that best reproduce the calibration measurements:

```python:
a_H, b_H, c_H = fitSemiempirical(alpha_H_calib, lmbda_calib, Z_calib, phi_H, D, E)
a_L, b_L, c_L = fitSemiempirical(alpha_L_calib, lmbda_calib, Z_calib, phi_L, D, E)
```

#### Step two: precompute the forward model:

```python:
tables = calcLookupTables(phi_H, phi_L, D, E, a_H, b_H, c_H, a_L, b_L, c_L, lmbda_range, Z_range)
lookup = Lookup(tables, lmbda_range, Z_range, interpolate_lmbda = True)
```

#### Step three: find the `{lambda, Z}` values which best reproduce the dual energy image:

```python:
lmbda, Z = calcZ(im_H, im_L, lookup, labels=labels)
```

#### Note that steps one and two only need to be performed once, and then the output can be saved to `tables.npy`. Step three can then be performed on several different images without needing to redo the calibration or recompute the forward model.

## References

#### For full algorithmic details and a validation using Geant4 Monte Carlo, see the original publication:

Peter Lalor, Areg Danagoulian. Atomic number estimation of dual energy cargo radiographs using a semiempirical transparency model. *Nuclear Instruments and Methods in Physics Research Section A: Accelerators, Spectrometers, Detectors and Associated Equipment*, 1064:169343, 2024. https://doi.org/10.1016/j.nima.2024.169343

## Questions?

For bugs, questions, or other inquiries, contact [Peter Lalor](mailto:plalor@mit.edu?subject=[GitHub]%20semiZ)
