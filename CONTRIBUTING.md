# Contributing to `drdg`

We welcome contributions to enhance reproducibility, add new clinical fundus imaging cohorts, or improve ordinal evaluation methodologies.

---

## Code of Conduct & Scientific Integrity

All contributions must adhere to high standards of scientific rigor:
1. **Neutral Academic Tone**: Avoid marketing buzzwords, unsubstantiated causal claims, or hype (e.g., avoid "champion", "superior", "state-of-the-art" without contextualized statistical evidence).
2. **Zero-Leakage Splitting**: Any new dataset integration must enforce strict patient-level isolation between training, validation, and testing partitions.
3. **No Hidden State**: All configuration must remain in strongly typed `@dataclass` structures (`drdg.config`). Avoid introducing YAML or Hydra configurations.
4. **Reproducibility**: All experiments must provide deterministic seed control and execute cleanly across both CPU and CUDA environments.

---

## Development Workflow

### 1. Environment Setup
```bash
git clone https://github.com/baohuy2209/deep-learning-for-computer-vision.git
cd deep-learning-for-computer-vision
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\Activate.ps1 on Windows
pip install -e .
pip install -r requirements.txt
```

### 2. Adding a New Cohort
To integrate an additional clinical fundus cohort:
1. Add an automated downloader in `src/drdg/data/downloaders/<cohort_name>.py`.
2. Add patient ID extraction regexes in `src/drdg/data/patient_ids.py`.
3. Register the dataset in `src/drdg/data/metadata.py`.
4. Add unit tests in `tests/test_patient_ids.py` and `tests/test_patient_splits.py`.

### 3. Running Tests Before Submitting PR
Ensure that all unit and integration tests pass:
```bash
pytest tests/ -v
```

Ensure notebooks have outputs stripped before committing:
```bash
# Strip outputs if modifying notebooks
python -c "import nbformat; [nbformat.write(nbformat.v4.new_notebook(), p) for p in []]"
```
