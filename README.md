# faultpick

Python toolkit for identifying and exploring clustered fault-related seismic patterns.

## Install and run

```bash
pip install -e .
faultpick --help
```

Or install dependencies from the requirements file into a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Dependencies

### Core

| Package | License | URL |
|---------|---------|-----|
| [NumPy](https://numpy.org) | BSD 3-Clause | https://numpy.org |
| [pandas](https://pandas.pydata.org) | BSD 3-Clause | https://pandas.pydata.org |
| [SciPy](https://scipy.org) | BSD 3-Clause | https://scipy.org |
| [scikit-learn](https://scikit-learn.org) | BSD 3-Clause | https://scikit-learn.org |
| [Pyrocko](https://pyrocko.org) | GPL-3.0-or-later | https://pyrocko.org |
| [glasbey](https://github.com/lmcinnes/glasbey) | MIT | https://github.com/lmcinnes/glasbey |

### Plotting (Not yet implemented)

| Package | License | URL |
|---------|---------|-----|
| [Matplotlib](https://matplotlib.org) | PSF | https://matplotlib.org |
| [Plotly](https://plotly.com/python/) | MIT | https://plotly.com/python/ |

### Development

| Package | License | URL |
|---------|---------|-----|
| [pytest](https://docs.pytest.org) | MIT | https://docs.pytest.org |

## License

This project is licensed under the GPL-3.0 License. See [LICENSE](LICENSE) for details.
