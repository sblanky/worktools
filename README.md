A number of tools I use for processing isotherms, that don't deserve their own repo yet.

# Installation
- Clone this repo
- `cd` into repo, then do `pip install .`

# Parsing and manipulating a pore size distribution form SAIEUS

There is a class for automatically parsing and manipulating the output from SAIEUS - `worktools.deconvolution.saieus`.

To import;

```py
from worktools.deconvolution import saieus

file = '/path/to/file.CSV'
dat = saieus.parse(file)
```

This stores all of the relevant data from the csv file. The data can be accessed by calling the relevant variable;

```py
print(dat.material)
``` 
	`$ ACC2600`

```py
print(dat.lambda_regularisation)
```
	`$ 4`

The pore size distribution and isotherm (with fit) can also be accessed via;

```py
print(dat.psd)
print(dat.isotherm)
```

Calculations can then be made on the data imported
