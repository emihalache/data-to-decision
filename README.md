# Data to Decision

This repository contains the code and data for the research project **"From Data to Decision: Investigating Bias Amplification in Decision-Making Algorithms"**. The project examines how biases present in the Adult/Census Income dataset influence the outputs of machine learning models, exploring whether these biases are amplified by Logistic Regression, Decision Tree, and Random Forest algorithms using various fairness metrics.

* **Title**: From Data to Decision: Investigating Bias Amplification in Decision-Making Algorithms
* **Date**: June 23, 2024
* **Institution**: Delft University of Technology
* **Link**: [Download the thesis (PDF)](https://repository.tudelft.nl/file/File_9d65fe61-d397-4adb-8129-d46a58f8250d?preview=1)

---

## Repository Structure

```
data-to-decision/
├── adult.csv
├── preprocess/
│   ├── preprocess_remove.py
│   ├── preprocess_impute_simple.py
│   ├── adult_preprocessed_train.csv
│   └── adult_preprocessed_test.csv
├── algorithms/
│   ├── algs.py
│   └── adult_trained_test_results.csv
├── bias/
│   ├── metrics_race.py
│   ├── metrics_sex.py
│   ├── metrics_plots.py
│   ├── race_plots.py
│   ├── sex_plots.py
│   └── dummy/
│       ├── data.csv
│       ├── pred.csv
│       └── metrics_dummy.py
└── README.md
```

* **preprocess/**: Scripts and preprocessed data for cleaning, encoding, and scaling the raw dataset. Note that `preprocess_impute_simple.py` is maintained for reference but **should not be used** in the analysis pipeline.
* **bias/**: Scripts to compute fairness metrics (Demographic Parity, Disparate Impact, Equal Opportunity, Equalized Odds) and generate corresponding plots.
* **algorithms/**: Contains `algs.py`, which trains Logistic Regression, Decision Tree, and Random Forest models on the preprocessed data, and outputs test set predictions and performance metrics to `adult_trained_test_results.csv`.

---

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/emihalache/data-to-decision.git
   cd data-to-decision
   ```

2. **(Optional) Create a virtual environment**:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:

   ```bash
   pip install pandas numpy scikit-learn fairlearn matplotlib seaborn
   ```

---

## Usage

1. **Data Preprocessing**

   * Run the preprocessing pipeline to clean, encode, and impute missing values:

     ```bash
     python preprocess/preprocess_remove.py
     ```
   * The processed training and test sets will be saved as `adult_preprocessed_train.csv` and `adult_preprocessed_test.csv` in the `preprocess/` directory.

2. **Model Training**

   * Train the classification models and generate test-set results:

     ```bash
     python algorithms/algs.py
     ```
   * The script will train Logistic Regression, Decision Tree, and Random Forest models, then save predictions and performance metrics to `algorithms/adult_trained_test_results.csv`.

3. **Bias Analysis**

      * Compute fairness metrics and generate plots:

      ```bash
       cd bias
      ```

      Then run:

      ```bash
       python metrics_race.py
       python race_plots.py
       python metrics_sex.py
       python sex_plots.py
      ```

      When finished, return to the root:

      ```bash
       cd ..
      ```


---

## Requirements

* Python 3.7 or higher
* pandas
* numpy
* scikit-learn
* fairlearn
* matplotlib
* seaborn

---

## Acknowledgements

* **Adult/Census Income Dataset**: UCI Machine Learning Repository

---

