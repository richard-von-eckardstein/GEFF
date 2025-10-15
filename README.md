# The gradient expansion formalism factory (GEFF)

This python package is designed to handle gauge-field production during cosmic inflation
using the gradient expansion formalism (GEF).

With this package, the user has access to built-in and ready-to-use models to determine the dynamical evolution during axion inflation. Furthermore, the GEFF package allows you to create your own versions
of inflationary models featuring gauge-field production, which can be solved using the GEF technique.

For more information on the GEFF, please see our [documentation](https://richard-von-eckardstein.github.io/GEFF/geff.html), and our [arXiv manual](https://arxiv.org/abs/2510.12644)

## Installation

To install this package use

```bash
pip install cosmo-geff
```

or use the `geff.yml` file to create a conda environment,

```bash
conda env create -f geff.yml
```

## Attribution

If you use this software in your work, please cite:

```
von Eckardstein, R. (2025). GEFF: The Gradient Expansion Formalism Factory. Zenodo. https://doi.org/10.5281/zenodo.17356579

@misc{von_eckardstein_2025_17356579,
  author       = {von Eckardstein, Richard},
  title        = {GEFF: The Gradient Expansion Formalism Factory},
  month        = oct,
  year         = 2025,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.17356579},
  url          = {https://doi.org/10.5281/zenodo.17356579},
}

von Eckardstein, R. (2025). GEFF: The Gradient Expansion Formalism Factory - A tool for inﬂationary gauge-ﬁeld production. arXiv. https://arxiv.org/abs/2510.12644

@misc{vonEckardstein:2025jug,
    author        = {von Eckardstein, Richard},
    title         = {GEFF: The Gradient Expansion Formalism Factory - A tool for inflationary gauge-field production},
    eprint        = {2510.12644},
    archivePrefix = {arXiv},
    primaryClass  = {astro-ph.CO},
    reportNumber  = {MS-TP-25-37},
    month         = oct,
    year          = 2025
}
```

