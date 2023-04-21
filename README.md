A number of tools I use for processing isotherms, that don't deserve their own repo yet.

# Installation
- Clone this repo
- `cd` into repo, then do `pip install .`

# Parsing and manipulating a pore size distribution form SAIEUS

There is a class for automatically parsing and manipulating the output from SAIEUS - `worktools.deconvolution.saieus`.

To import a SAIEUS file;

```py
from worktools.deconvolution import saieus

file = '/path/to/file.CSV'
dat = saieus.parse(file)
```

This stores all of the relevant data from the csv file. The data can be accessed by calling the relevant variable;

```py
print(dat.material) # outputs material id, e.g. ACC2600
``` 

```py
print(dat.lambda_regularisation) # outputs lambda used in regularisation, e.g. 4
```

The pore size distribution and isotherm (with fit) can also be accessed via;

```py
print(dat.psd)
print(dat.isotherm)
```

Calculations can then be made on the data imported, for example;

```py
print(dat.peak()) # peak of pore size distribution
print(dat.porosity_slice()) # get pore volume, surface area between any two pore widths
print(dat.pore_region_slice('micro')) # get pore volume, surface area in micropore region.
```

# Current status

Work in progress!
