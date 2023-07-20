# decomp

This code takes Eagle stellar particle data as input and, for a specified list of GroupNumbers, allocates the particles in that group to either the disc, bulge or IHL component.

By default, the code outputs particle allocations individually for each group (central_XX.pkl), along with a summary file with component mass fractions for each group (summary.pkl).

To following code will decompose the stellar component of 3 central galaxies with GroupNumbers 100, 666, 1000:

```
python main.py -gl [100, 666, 1000] -b "/path/to/eagle/data/" -o "/path/to/output/folder/"
```
