# AutoCoevoRUL

This repository holds the code for our paper "Coevolution of Remaining Useful Lifetime Estimation Pipelines for Automated Predictive Maintenance" by Tanja Tornede, Alexander Tornede, Marcel Wever and Eyke Hüllermeier. Regarding questions please contact tanja.tornede@upb.de .


Please cite this work as

```
@inproceedings{tornede2021coevolution,
  title={Coevolution of Remaining Useful Lifetime Estimation Pipelines for Automated Predictive Maintenance},
  author={Tornede, Tanja and Tornede, Alexander and Wever, Marcel and H{\"u}llermeier, Eyke},
  journal={Proceedings of the Genetic and Evolutionary Computation Conference},
  year={2021}
}
```


## Abstract
Automated machine learning (AutoML) strives for automatically constructing and configuring compositions of machine learning algorithms, called pipelines, with the goal to optimize a suitable performance measure on a concrete learning task. So far, most AutoML tools are focused on standard problem classes, such as classification and regression. In the field of predictive maintenance, especially the estimation of remaining useful lifetime (RUL), the task of AutoML becomes more complex. In particular, a good feature representation for multivariate sensor data is essential to achieve good performance. Due to the need for methods generating feature representations, the search space of candidate pipelines enlarges. Moreover, the runtime of a single pipeline increases substantially. In this paper, we tackle these problems by partitioning the search space into two sub-spaces, one for feature extraction methods and one for regression methods, and employ cooperative coevolution for searching a good combination. Thereby, we benefit from the fact that the generated feature representations can be cached, whence the evaluation of multiple regressors based on the same feature representation speeds up, allowing the evaluation of more candidate pipelines. Experimentally, we show that our coevolutionary strategy performs superior to the baselines.


## Execution Details (getting the code to run)
For the sake of reproducibility, we will detail how to reproduce the results presented in the paper below. First, clone this repository. 


### 1. Setup Database Configuration
We assume you have a MySQL server with version >= 5.7.9 running. In the cloned project you will find a configuration folder entitled `conf`. Create database configuration files for each of the approaches you want to execute: 
* AutoCoevoRUL: `conf/experiments/coevolution.properties`
* ML-Plan-RUL: `conf/experiments/mlPlan.properties`
* Random Search: `conf/experiments/randomSearch.properties`

Each configuration file should contain the following information: 

```
db.driver = mysql
db.host = {URL}
db.username = {USERNAME}
db.password = {PASSWORD}
db.database = {DATABASE_NAME}
db.table = {TABLE_NAME}
db.ssl = {TRUE/FALSE}
```

You have to adapt all entries according to your database server setup. The entries have the following meaning:
* `host`: the address of your database server
* `username`: the username the code can use to access the database
* `password`: the password the code can use to access the database
* `database`: the name of the database where tables will be created
* `table`: the name of the table, where results should be stored. This is created automatically by the code if it does not exist yet and should NOT be created manually.
* `ssl`: whether ssl should be used or not



### 2. Setup Environment
Singularity has to be installed, please follow the according instructions from their [installation guide](https://sylabs.io/guides/3.0/user-guide/installation.html). Note that the `python_connection` dependency is the one you find on the top level of this repository. Next you can build the container from the definition file which you can find in this project via

```
sudo singularity build AutoCoevoRUL.sif conf/cluster/singularity.recipe
```

Alternatively you can set up an [Anaconda](https://anaconda.org/) environment with the according dependencies. If you decided to do so, you have to adapt the python configuration `conf/python.properties` according to your environment. You have the following entry options to set: 
* `pythonCmd`
* `pathToPythonExecutable`
* `anaconda`
* `pathToCondaExecutablepathToCondaExecutable`



### 3. Setup Experiment Configuration
In the experiment configuration folder of the cloned project you will find the configuration file `conf/experiments/experiments.cnf`. Reproducing the experiments means using the exact same configuration than we did to achieve the results of the paper. If you make changes in this file, you will get different results according to your configuration. 

Depending on the computational resources you will run the experiments on, you might have to change the entries `cpu.max` and `mem.max` accordingly. 


### 4. Data
The truncated data we used for the experiments can be found in the repository folder named `data`. Make sure to have that cloned to your deviceas well. You may have to adapt the root path of the data folder in the above mentioned experiment configuration file. The entry that has to be changed is named `dataPath`. 


### 5. Obtaining Evaluation Results
At this point you should be good to go and can execute the experiments as explained in the following:
* For AutoCoevoRUL you have to execute the `src/main/java/autocoevorul/AutoCoevolutionRunner`
* For ML-Plan-RUL you have to execute the `src/main/java/autocoevorul/baseline/mlplan/MLPlanRunner`
* For Random Search you have to execute the `src/main/java/autocoevorul/baseline/randomsearch/RandomSearchRunner`

Using the following parameters for each class will execute the experiments: `MyPC true true`, which can be adapted for more specific needs according to:
1. Execution host name (e.g. `MyPC`)
2. Boolean if the database has to be set up
3. Boolean if the code should be executed

It is important to first setup the database before executing the code. Of course, this can be done in a single execution, if both booleans are set to `true`.

After the execution you will find you database table filled with the results. Per default the total search time is set to 4 hours, additionally considering the final evaluation afterwards, it could take up to 4.5 hours until the results are available. For each approach the asymmetric loss can be found in the according column called `performance_asymmetric_loss`. The major configuration to distinguish different runs is denoted in the first columns.


### 6. Generating Results Tables
If you want to re-generate the tables presented in the paper, navigate into the project root folder. There you will find a gradle wrapper file. In order to generate LaTeX tables, run 
* On Windows: ```.\gradlew generateTables```
* On Mac OS: ```./gradlew generateTables```

The LaTeX table will be printed to the command line. 
