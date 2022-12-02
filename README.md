# How Predictable is Traffic Congestion

- This is the source code of the paper manuscript: **<How predictable are macroscopic traffic states: a perspective of
uncertainty quantification>**

- The manuscript is still under review.

## Quick start

### Requirements:

* Python = 3.9
* tensorflow >= 2.10.0
* Keras >= 2.10.0
* Shapely = 1.8.5

### Data Preparation

* The used data is collected and prepared by NDW [National Data Warehouse](https://interaction-dataset.com/). 
* Send us an email: [G.Li-5@tudelft.nl](G.Li-5@tudelft.nl). We will share the fully-processed data that is ready to use.
* Put the datasets in the `src_code/data/` folder.

### Model Training

* Run the `ModelTraining.ipynb` to train the deep ensembles of forecasting models.
* Detailed instructions are provided in the notebook.

### Paper Reproduce

* Run the `AnalysisScript.ipynb` to get the quantified uncetainty, predictions, etc.
* Detailed instructions are provided in the notebook.
