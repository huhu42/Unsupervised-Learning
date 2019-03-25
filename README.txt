Unsupervised Learning and Dimensionality Reduction
Liyue (Nikki) Hu - lhu81

This link contains all information needed to run this assignment: 

Requirements: 
You will need to use python 3 with this code, and to pip install the packages in `requirements.txt`. The main addition here is the tables module which _does_ require HDF5. If you are using OS X with Homebrew you can simply `brew install hdf5` before installing the requirements. 
If this does not work for you, try the `requirements-no-tables.txt` file. Windows users have noted the need to install the tables module but on some systems this is not required. 

Overall Flow
1. Run the various experiments via `python run_experiment.py --all`
2. Plot the results so far via `python run_experiment.py --plot`
3. Run `run_clustering.sh`, the dim values have been set
4. One final run to plot the rest `python run_experiment.py --plot`
5. Run `consolidate_nn_data_clean.py` to get csv of all the key performance metrics

Output
Output CSVs and images are written to `./output` and `./output/images` respectively. Sub-folders will be created for each DR algorithm (ICA, PCA, etc) as well as the benchmark.

If these folders do not exist the experiments module will attempt to create them.

