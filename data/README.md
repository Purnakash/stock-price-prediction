# data/

Optional CSV files for `--csv` / the app's "Upload CSV" option.

Any CSV with a `Date` and a `Close` column works; Open, High, Low and Volume
are used when present. NSE/BSE exports are recognised as-is (for example
`Total Trade Quantity` is read as volume).

The original project used `NSE-TATA.csv` (Tata Global Beverages, NSE, 2013–2018).
The same company now trades as **TATACONSUM.NS**, so the app can download its
history directly. CSV files here are not committed (see `.gitignore`).
