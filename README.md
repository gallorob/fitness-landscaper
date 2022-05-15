# fitness-landscaper: A simple 2D fitness landscapes generator.

This library can be used to define 2D landscape or generate them randomly using a multitude of components.

Properties:
- Domains are bounded and the fitness values can be bounded within any desired range
- Components can be combined by taking the max of their values or by summing them together
- Components are weighted, so it's easy to generate landscape with both peaks (positive weight) and sinks (negative weight)
- Final fitness value can be disturbed with Gaussian noise to simulate noisy landscapes

## How to install

1. Make sure you have all requirements installed (run `pip install -r requirements.txt` to check and install missing packages)
2. Install the library with `pip install .`

## How to use

See `example.ipynb`.

## Cite

If you use this library for a project, let me know!

If you need a reference for this project, use the following `bibtex`:
```
@misc{fitland-2022-git,
    author = {Gallotta, Roberto},
    title = {{fitness-landscaper}: A simple 2D fitness landscapes generator},
    year = {2022},
    publisher = {GitHub},
    journal = {GitHub repository}}
}
```