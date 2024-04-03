# Interpretable Models
This repository contains code for several interpretable models:

1. **SIRUS**
2. **L0 Rulefit**
3. **Simply Rules**

## Model Descriptions 
### 1. SIRUS

- **Important Parameters**: 
    - `threshold` (int): Determines the frequency at which a rule must occur to be used (accepted) in the linear model.

- **Description**:
    1. Quantize the dataset (optional).
    2. Fit a random forest with a depth of 2.
    3. Extract rules from the trees.
    4. Accept rules which have a frequency greater than the provided threshold.
    5. Fit a linear model on the rules with ridge regularization.

    The implementation provides an option to not quantize the dataset by specifying a `FilterType`:
    - `FilterType(1)`: Quantizes the dataset, where the split value of each split comes from the quantile.
    - `FilterType(2)`: Non-quantizes the dataset, where the split value of each split comes from each dataset sample.

    Note: In SIRUS, you cannot directly control the number of rules used in the linear model; it is instead controlled by the `threshold` parameter.

- **Sources**: [https://proceedings.mlr.press/v130/benard21a.html](https://proceedings.mlr.press/v130/benard21a.html)

### 2. L0 Rulefit

- **Important Parameters**:
    - `Max rules` (int): Determines the maximum rules used in the linear model. The maximum rules are controlled by L0 regularization.

- **Description**:
    1. Quantize the dataset (optional).
    2. Fit a random forest with a depth of 2.
    3. Extract all rules.
    4. Fit a linear model over the rules with L0 regularization (using L0Learn).

    Note: In L0 Rulefit, you can control the number of rules used in the linear model by the `max rules` parameter, which is a hyperparameter in L0 regularization.

- **Sources**: [https://arxiv.org/abs/2202.04820](https://arxiv.org/abs/2202.04820)

### 3. Simply Rules

- **Description**:
    1. Fit a random forest with a depth of 2.
    2. Extract all rules.
    3. Fit a linear model over the rules without regularization.

    Note: This model was implemented with the purpose of generating rules without a specific aim towards predictive power.

## Project structure 

```
├── data
│   ├── boston_housing.csv
│   ├── BreastWisconsin.sh
├── src
│   ├── L0_rulefit
│   │   ├── L0rulefit.py
│   ├── rule_generator
│   │   ├── simply_rules.py
│   ├── sirus
│   │   ├── dependent.py
│   │   ├── sirus.py
│   ├── data.py
│   ├── forest.py
│   ├── linear.py
│   ├── quntiles.py
│   ├── ruleEsemble.py
│   ├── rules.py
│   ├── tree.py
├── tests
│   ├── unitests.py
│   ├── utils.py
├── .gitignore
├── README.md
├── requirements.txt
```

* `src/L0_rulefit/L0rulefit.py` : Contains the class for the model L0_Rulefit.
* `src/rule_generator/simply_rules.py` : Contains the class for the model SimplyRules.
* `src/sirus/dependent.py` : Contains the code to identify and remove dependent rules used in the SIRUS model.
* `src/sirus/sirus.py`: Contains the class for the model SIRUS.
* `src/data.py` : Contains an enum called DataType used to distinguish between Classification and Regression problems. Also contains functions to load datasets.
* `src/forest.py` : Contains a random forest class that can fit a tree based on split values that come from the quantiles.
* `src/linear.py` : Contains an enum called RegType used to define the regularization type. Also contains functions to fit linear model with different regularizations.
* `src/quantiles.py` : Contains code to quantize the dataset.
* `src/RuleEsemble.py` : Contains code to extract rules from a random forest. The random forest can be an instance of Sklearn's or the class defined in `src/forest.py`.
* `src/rules.py` : Contains code to define a rule and helper functions to deal with rules.
* `src/tree.py` : Contains a class to build a tree where each split value can be based on a quantile.


## How to use models

### SIRUS

1. **Define Parameters**: 
   - `filterType(1)` indicates that the dataset is quantized, and two rules are classified as the same if they are both split on the same feature and have the same split value.

```python 
NUM_TREES = 100
MAX_DEPTH = 2
P0 = 0.080
MIN_DATA_LEAF = 5
PARTIAL_SAMPLING = 0.7
NUM_QUANTILES = 10
SEED = 4
filter_type = FilterType(1)  
```

2. **Instantiate an Instance of the Sirus Class**: 
   - The Sirus model has separate classes for Regression and Classification.

```python
model = SirusRegression(
    threshold=P0,
    max_depth=MAX_DEPTH,
    min_samples_leaf=MIN_DATA_LEAF,
    num_trees=NUM_TREES,
    partial_sampling=PARTIAL_SAMPLING,
    quantiles=splits,
    max_split_candidates=NUM_FEATURES,
    filter_type=filter_type,
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

### L0 Rulefit

1. **Define Parameters**: 
   - In L0 Rulefit, there are no different classes for Regression and Classification. Instead, you must use the enum `DataType` to define the data type.

```python
MAX_DEPTH = 2
NUM_QUANTILES = 10
MIN_LEAF_DATA = 5
SEED = 2
SAMPLE_FRAC = 0.70
NUM_TREES = 100
MTRY = 1 / 3
MAX_NUM_RULES = 10
data_type = DataType.CLASSIFICATION
```

2. **Instantiate the L0 Rulefit Model and Fit the Model**. 

```python
model = L0_Rulefit(
    data_type=data_type,
    max_depth=MAX_DEPTH,
    partial_sampling=SAMPLE_FRAC,
    min_samples_leaf=MIN_LEAF_DATA,
    num_trees=NUM_TREES,
    max_rules=MAX_NUM_RULES,
    max_split_candidates=MAX_SPLIT_CANDIDATES,
    random_state=SEED,
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```