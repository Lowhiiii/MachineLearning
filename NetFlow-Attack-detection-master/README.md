Network Anomaly Detection
-------------------------

Detecting attacks on IP network using Node2Vec embedding representation and supervised classification.

# Instructions 

## Setup

----
1. **Download data**
- Download data file from the following link 
 https://rdm.uq.edu.au/files/8c6e2a00-ef9c-11ed-827d-e762de186848
- Create the directory `data/raw`
- Extract `NF-UNSW-NB15-v2.csv` file to `data/raw` directory

2. **Install requirmeents**
- Recommended : Use conda or virtualenv to create a new environment (optional)
- From the root directory of the project, run the following command to install the depencencies
```bash
  pip install -r pkg/requirements.txt
```
 
 ####  Note : How to setup pyenv
 
   ``pyenv install 3.10.5``

   ``pyenv virtualenv 3.10.5 network_ad_env``
   
   ``cd [PROJECT ROOT DIRECTORY]``

   ``pyenv local network_ad_env``
  
  

## Exploratory Data Analysis (EDA)

- Complete the notebook notebooks/eda.ipynb




