├── run.py			| Run source files
|-------------------------------|
├── data/			| Contains generated data
│├── archive/			| Scripts no longer used
│├── datasets/			| Generated datasets (not synced to git)
│├── test/			| Tests to verify data
|-------------------------------|
├── models/			| Contains models 
│├── figures/			| Error distribution of trained models
│├── history/			| Training history of trained models
│├── logs/			| Logs (MSE, structure) of trained models
│├── models/			| Saved tensorflow models
|-------------------------------|
├── src/			| Source files
│├── archive/			| Scripts no longer used
│├── execution.py		| Take care of multiprocessing when calling data generation/machine learning
||------------------------------|
│├── d/				| Data generation and processing
││├── data_generation.py	| Data generation
││├── data_transformation.py	| Executes data fetching and pre-processing
││├── transformators.py		| Scikit-Pipeline transformators used in data_transformation.py
││├── utils.py			| Small functions used in data generation
||------------------------------|
│├── ml/			| Machine learning
││├── building.py		| Build deep networks
││├── machine_learning.py	| Execute machine learning (building, training, validation)
││├── training.py		| Train deep networks
││├── utils.py			| Small functions used in machine learning
││├── validation.py		| Validation of deep networks, graph plotting

