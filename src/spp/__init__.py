"""Stock price prediction with an LSTM: data, features, model, metrics, forecast."""

import warnings

# yfinance still uses pandas options that pandas 3.x deprecates (e.g. inplace=True).
# They are harmless for this project and would only clutter the console.
for _category in (DeprecationWarning, FutureWarning):
    warnings.filterwarnings("ignore", category=_category, module=r"yfinance(\..*)?$")
warnings.filterwarnings("ignore", message=r".*inplace keyword.*is deprecated.*")

__version__ = "2.0.0"
