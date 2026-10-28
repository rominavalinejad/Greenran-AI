# Data license

**The MIT license in [`LICENSE`](LICENSE) covers the code only.** The data in this repository is derived from a
third-party database and is made available under the ODbL 1.0, as described below.

## Source and attribution

The data is derived from the Telecom Italia Big Data Challenge dataset
*Telecommunications - SMS, Call, Internet - MI* (Milan), published on Harvard Dataverse,
[doi:10.7910/DVN/EGZHFV](https://doi.org/10.7910/DVN/EGZHFV)
([dataset page](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EGZHFV)),
under the Open Database License (ODbL) 1.0: <https://opendatacommons.org/licenses/odbl/1-0/>.

The data provider requires this attribution: use of the data must be accompanied by the hyperlink
[from BigDataChallenge contest](http://www.telecomitalia.com/tit/en/bigdatachallenge.html)
(see also the [dataset page](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EGZHFV)).

## What is covered (ODbL 1.0, share-alike)

The following files are derived databases and are made available under the ODbL 1.0. Under its share-alike term,
adapted versions must also be shared under the ODbL, with the attribution above.

- `data/raw/` (one parquet file per day, 2013-11-04 to 2013-12-01)
- `data/processed/` (`site_load.parquet`)
- the decisions parquet files in `results/` and `results_v6/` (`decisions_*.parquet`), which contain the hourly site load
  from the test window together with forecasts and sleep decisions

## How the data was processed

Verified against `download_data.py`, `config.json` and `src/greenran/` (`data.py`, `topology.py`):

1. **Squares:** from the 100x100 Milan grid, a 20x20 block of grid squares (rows 45-64, columns 40-59, 400 squares) is
   kept. The daily files are filtered to these squares while streaming, so full daily files are not stored.
2. **Days:** 28 days, 2013-11-04 to 2013-12-01.
3. **Hourly aggregation:** the 10-minute records are summed over country codes and aggregated to hourly totals per square
   (internet, calls = call_in + call_out, sms = sms_in + sms_out); and timestamps are converted to Europe/Rome local time.
   This gives `data/raw/<day>.parquet`. When the hourly series is assembled for modelling, any missing hours are filled by
   interpolation per square.
4. **Synthetic sites:** each 2x2 block of squares is treated as one simulated site (100 sites), and the site load is the
   sum of its squares' internet activity (`data/processed/site_load.parquet`). The site and cell layout and capacities
   are synthetic assumptions of this project; grid-square activity is only a proxy for cell load.

No other data source is used.
