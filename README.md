# Interpretable_Models

This respository contains code for numerous interpretable models 
1. SIRUS
2. L0_Rulefit 

SIRUS: 

## Project structure 

```
├── data
│   ├── boston_housing.csv
│   ├── BreastWisconsin.sh
├── src
│   ├── L0_rulefit
│   │   ├── L0rulefit.py
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
```

* `src/L0_rulefit/L0rulefit.py` : This file contains the class for the model L0 rulefit 
* `src/sirus/sirus.py` : This file contains the class for the model SIRUS
