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
from percept import PerceptParser
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
