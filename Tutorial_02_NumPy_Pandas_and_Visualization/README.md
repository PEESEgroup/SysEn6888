# Tutorial 02 NumPy, Pandas, and Visualization

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PEESEgroup/SysEn6888/blob/main/Tutorial_02_NumPy_Pandas_and_Visualization/tutorial_02_data.ipynb)

NumPy and Pandas foundations, a complete energy-data analysis, Matplotlib styling, Seaborn statistical graphics, and interactive Plotly charts and maps.

**Runtime:** CPU. The energy data are embedded; the tips and county-map examples download public historical datasets automatically. Internet access is required. Open the notebook, connect, and select **Runtime → Run all**. Dependency setup is included. Each notebook runs independently without local files or earlier lessons.

Basic Python is helpful. The cleaning example removes one duplicate and two rows with missing/invalid energy readings, leaving ten observations totaling 222 kWh. Outputs include a cleaned CSV, plots, a self-contained interactive HTML chart, and `outputs/tutorial_02/metrics.json`. All energy values are synthetic teaching data.

The notebook includes 24 code cells. Allow 90–120 minutes for reading and exercises; execution is much shorter. You can also download the notebook and use **File → Upload notebook** in Colab. See the [series README](../README.md) and [validation report](../docs/validation.md).

See the notebook for data references and the repository LICENSE for license information.

## Topics

| Sections | Content |
| --- | --- |
| 2–3 | Arrays from lists, zeros/ones/identity, ranges, random distributions, shape/dtype, reshape, extrema, indexing, slices, assignment, views/copies, fancy indexing, broadcasting, arithmetic, numerical edge cases, dot products, inverse, norms, and linear solves |
| 4 | Series construction and label alignment; DataFrame selection, editing, dropping, masking, index resets, and MultiIndex cross-sections |
| 5–8 | Data cleaning audit, group summaries, pivot/merge, static energy plots, interactive scatter, and CSV/HTML exports |
| 9 | Pyplot and object-oriented figures, subplots/insets, named and hexadecimal colors, CSS4/XKCD comparison, widths, dashes, markers, and text annotations |
| 10 | Tips dataset: relational facets, histogram/KDE, swarm, split violin, and bar plots |
| 11–12 | County unemployment choropleth, FIPS join checks, downloaded-data hashes, and final output verification |

Intentional indexing errors are caught and explained, so **Run all** continues. Energy measurements are synthetic; the restaurant and 2016 unemployment examples are historical observations. The map is not current economic data. Map HTML may need internet access for its geographic basemap.
