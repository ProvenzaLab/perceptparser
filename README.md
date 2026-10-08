# 🧠 Percept Parser

The **Percept Parser** is a lightweight Python package designed to extract and organize data from Medtronic Percept™ JSON session reports. It simplifies downstream analysis by converting complex report structures into structured directories.

---

## 🚀 Installation

We recommend using [**uv**](https://github.com/astral-sh/uv) for fast, isolated, and reproducible Python environments.

```bash
# Create and activate a virtual environment
uv venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install the package
uv pip install percept-parser
```

Basic Usage:

```python
from percept_parser.percept import PerceptParser
import os

# Define your session report JSON file
filename = "Report_Json_Session_Report_20250425T080622.json"

# Initialize the parser
parser = PerceptParser(filename)

# Use the filename (without .json) as output directory name
dir_name = os.path.basename(filename)[:-len(".json")]

# Parse all content and save it to the specified directory
parser.parse_all(out_path=dir_name)
```

For pulsatile artifacts in time-domain recordings, select one of two independent
options: `clean_pulsatile_artifacts=True` masks detected intervals with `NaN`,
or `subtract_pulsatile_artifacts=True` subtracts an amplitude-fitted pulse
template from the detected intervals while retaining finite samples. The two
options are mutually exclusive and can each be combined with either ECG
removal method.

Both modes use a PulsAr-inspired detector: a 0.5–30 Hz filter, an
autocorrelation RMS threshold of 0.081 measured over lags from 1–10 seconds,
50–120 bpm pulse intervals with at most 100 ms median absolute deviation, and
template cross-correlation. Autocorrelation peaks are searched out to 30
seconds (or half the segment length when shorter) to make the periodicity
estimate more robust when longer recordings are available. In subtraction mode,
the template is averaged from matched events in each channel and fitted
independently to each event. Segments shorter than 20 seconds are left unchanged. In both pulsatile modes the first 10 seconds of every time-domain dataframe are set to NaN (start-up transient) before detection.

```python
parser = PerceptParser(
    filename,
    ecg_method="template",
    subtract_pulsatile_artifacts=True,
)
parser.parse_all(out_path=dir_name, plot=True)
```
