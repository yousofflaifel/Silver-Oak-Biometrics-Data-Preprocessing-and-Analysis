# Silver Oak Biometrics Data Preprocessing and Analysis

A Python data preprocessing and exploratory analysis project using a **Silver Oak biometrics dataset**. The project focuses on improving data quality through duplicate removal, invalid-value handling, missing-value imputation, categorical cleaning, feature engineering, discretization, encoding, scaling, and time-based analysis.

The work is implemented in a Jupyter Notebook using Pandas, NumPy, Matplotlib, Seaborn, and selected Scikit-learn preprocessing tools.

## Project Overview

The project starts with a raw CSV dataset and applies a structured preprocessing workflow before performing further analysis.

The main workflow is:

```text
Raw CSV Dataset
      |
      v
Initial Inspection
      |
      v
Duplicate Removal
      |
      v
Invalid Value Detection
      |
      v
Missing Value Analysis
      |
      v
Logical / Group-Based Imputation
      |
      v
Categorical Data Cleaning
      |
      v
Feature Engineering
      |
      v
Discretization
      |
      v
Categorical Encoding
      |
      v
Normalization / Standardization
      |
      v
Time-Based Analysis
```

## Technologies Used

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

Scikit-learn is used for preprocessing techniques including:

- `LabelEncoder`
- `MinMaxScaler`
- `StandardScaler`

## Dataset

The notebook loads the dataset from:

```text
Biometrics Data of Silver Oak.csv
```

The data contains measurements related to Silver Oak logs, including variables such as:

- Length
- Diameter
- Wet Weight
- Dry Weight
- Moisture
- Log Volume
- Dry-to-Wet Weight Ratio
- Altitude
- Slope
- Measurement Date
- Log
- Part
- Layer
- Tree Group
- Region
- Health Status
- Soil Type
- Exposure
- Diameter Class
- Moisture Class

## Data Loading and Initial Inspection

The project begins by loading the CSV file using Pandas:

```python
df_raw = pd.read_csv("Biometrics Data of Silver Oak.csv")
```

Initial inspection includes:

- First and last rows
- Dataset shape
- Data types
- Descriptive statistics
- Selected columns and records
- Identification of potentially unusual values

The raw dataset is retained so that the original data can be compared with the cleaned version.

## Duplicate Detection and Removal

Duplicate observations are identified using Pandas:

```python
df_raw.duplicated().sum()
```

Duplicates are removed into a separate cleaned DataFrame:

```python
df_clean = df_raw.drop_duplicates().copy()
```

Keeping the raw DataFrame preserves the original dataset for comparison and validation.

## Data Cleaning

### Column Names

Column-name spelling issues are corrected to make the dataset easier to understand and work with.

For example:

```text
Leght_m → Length_m
Dai_cm  → Diameter_cm
```

### Missing Value Standardization

Different representations of missing data are converted to `NaN`, including:

```text
""
" "
"NA"
"N/A"
"na"
"null"
"-"
```

### Invalid and Out-of-Range Values

Values that are physically or practically invalid are converted to missing values.

Examples include:

- Length less than or equal to zero
- Diameter less than or equal to zero
- Wet or dry weight less than or equal to zero
- Non-positive log volume
- Negative moisture
- Invalid dry-to-wet weight ratios
- Negative altitude
- Slope values outside the range of 0 to 90 degrees

This approach avoids replacing invalid observations with arbitrary values while keeping the data-quality issue visible for later treatment.

## Missing Value Analysis

Missing values are counted and visualized before imputation.

A correlation matrix and heatmap are also created to investigate relationships between numerical variables and help determine suitable approaches for filling missing values.

## Group-Based Missing Value Imputation

Several missing numerical variables are filled using grouped median values.

### Diameter

`Diameter_cm` is filled using the median diameter within groups created from quartiles of `Ww_kg`.

The process uses:

```python
pd.qcut()
groupby()
transform()
```

### Log Volume

Missing `LogVolume_m3` values are filled using grouped medians based on quartile groups of `Diameter_cm`.

### Length

Missing `Length_m` values are filled using grouped medians based on quartile groups of `LogVolume_m3`.

Using `transform()` keeps the resulting values aligned with the original DataFrame.

## Formula-Based Missing Values

Some variables are derived from other measurements.

### Moisture Percentage

The project calculates moisture using:

```text
%Moisture = ((Wet Weight - Dry Weight) / Dry Weight) × 100
```

Implemented as:

```python
df_clean["%Moisture"] = (
    (df_clean["Ww_kg"] - df_clean["Dw_kg"]) / df_clean["Dw_kg"]
) * 100
```

### Dry-to-Wet Weight Ratio

The ratio is calculated as:

```text
DwToWwRatio = Dry Weight / Wet Weight
```

Implemented as:

```python
df_clean["DwToWwRatio"] = df_clean["Dw_kg"] / df_clean["Ww_kg"]
```

## Derived Categories

### DiameterClass

Diameter is converted into fixed-width categories using `pd.cut()`:

```text
Very Small
Small
Medium
Large
```

### MoistureClass

Moisture percentage is categorized into:

```text
Low
Medium
High
```

### Data-Quality Correction

After deriving `MoistureClass`, an invalid record with negative moisture was identified because the dry weight exceeded the wet weight. That row was removed from the cleaned dataset.

## Data Types and Categorical Cleaning

The project converts appropriate variables to suitable data types.

The measurement date is converted to a datetime object:

```python
df_clean["MeasurementDate"] = pd.to_datetime(
    df_clean["MeasurementDate"],
    errors="coerce"
)
```

Categorical variables are converted to Pandas categorical data types.

### Log Cleaning

The `Log` column is cleaned by:

- Removing extra spaces
- Standardizing separators
- Converting numeric-only identifiers into a consistent `LOG_` format
- Setting the column as categorical

### Part Cleaning

The `Part` column is cleaned by:

- Removing extra spaces
- Standardizing capitalization
- Correcting `Brance` to `Branch`

## Feature Engineering

Three new features are created.

### ThinnessRatio

```text
ThinnessRatio = Length_m / Diameter_cm
```

This represents the length of a log relative to its diameter and provides additional information about its geometric form.

### RegionAvgMoisture

The average moisture percentage is calculated for each region and mapped back to individual records.

This adds regional environmental context to each log.

### TerrainType

Slope is converted into three terrain categories:

```text
Gentle
Moderate
Steep
```

## Discretization

The project compares two approaches to discretization:

### `cut`

Creates fixed-width intervals.

It is used where domain interpretability and fixed numeric ranges are important.

### `qcut`

Creates approximately equal-frequency bins.

It is used during grouped median imputation to divide observations into quartile-based groups.

## Categorical Encoding

Different encoding techniques are demonstrated depending on the type of categorical variable.

### One-Hot Encoding

Nominal variables without an inherent order are converted using:

```python
pd.get_dummies()
```

This is applied to variables such as:

- Part
- TreeGroup
- Region
- SoilType
- Exposure

### Manual Ordinal Encoding

`HealthStatus` is manually mapped according to its order:

```text
Healthy  → 1
Moderate → 2
Stressed → 3
```

### Label Encoding

`LabelEncoder` is used for categorical classes such as:

- DiameterClass
- MoistureClass

## Scaling

Two numerical scaling techniques are demonstrated.

### Min-Max Normalization

`MinMaxScaler` scales numerical variables to a range between 0 and 1.

```python
from sklearn.preprocessing import MinMaxScaler
```

### Standardization

`StandardScaler` transforms variables into a centered distribution with unit variance.

```python
from sklearn.preprocessing import StandardScaler
```

The project compares normalization and standardization and notes that extreme moisture values make standardization more suitable for the analysis.

## Exploratory Analysis and Visualization

The notebook uses Matplotlib and Seaborn for visualization.

Examples include:

- Missing-value bar charts
- Correlation heatmap
- Diameter distribution histogram
- Diameter class pie chart
- Moisture analysis
- Average moisture over time

## Average Moisture Over Time

The project calculates the average moisture percentage by measurement date:

```python
daily_moisture = df_clean.groupby(
    "MeasurementDate"
)["%Moisture"].mean()
```

A line plot is then used to examine changes in average moisture over time.

The analysis found that moisture was relatively stable during much of the measured period, with occasional larger spikes.

## Additional Exploration

The notebook also includes an additional filtering example that selects records where:

```text
HealthStatus = Healthy
Slope_deg >= 10
```

It then examines diameter-class counts and selected columns and checks missing values.

## How to Run

### Requirements

Install Python and the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Steps

1. Clone or download this repository.
2. Place the dataset file in the same directory as the notebook.
3. Make sure the dataset is named:

```text
Biometrics Data of Silver Oak.csv
```

4. Open Jupyter Notebook:

```bash
jupyter notebook
```

5. Open:

```text
FinalProject.ipynb
```

6. Run the notebook cells from top to bottom.

The notebook expects the CSV file to be available locally because it loads the dataset using:

```python
pd.read_csv("Biometrics Data of Silver Oak.csv")
```

## Project Files

The main project file is:

```text
FinalProject.ipynb
```

The notebook contains the complete Python implementation of the preprocessing and analysis workflow.

The dataset used by the notebook is:

```text
Biometrics Data of Silver Oak.csv
```

The accompanying report documents the decisions and justification behind the preprocessing steps.

## Learning Outcomes

This project demonstrates practical experience with:

- Data loading
- Data inspection
- Data quality assessment
- Duplicate detection and removal
- Invalid-value handling
- Missing-value analysis
- Group-based imputation
- Formula-based feature completion
- Categorical data cleaning
- Feature engineering
- Discretization
- One-hot encoding
- Ordinal encoding
- Label encoding
- Normalization
- Standardization
- Correlation analysis
- Data visualization
- Time-series exploration
- Pandas DataFrame operations
- Scikit-learn preprocessing

## Author

**Yousof Flaifel**

Data Science & AI Student  
Al Hussein Technical University (HTU)

---

University Data Science and Preprocessing Project
