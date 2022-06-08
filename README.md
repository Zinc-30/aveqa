# AVEQA
the reproduction code of the paper "Learning to Extract Attribute Value from Product via Question Answering: A Multi-task Approach"

# Usage
The model would run in 1 GPU in defualt, to change the setting, change the number in **find_gpus** in **train.py and test.py**


Put the aepub dataset in the **./dataset** folder as an txt file (./dataset/publish_data.txt), and the processed dataset would be save to **'./dataset/aePub'**


Use the **ae_pub.py** to generate and preprocess the dataset.


Modify the **config.json** to set the parameter and the dataset path.


The **training.py** would automatically go through the training and testing pipeline and generate the training and testing dataset

After training, the training accuracy would be stored in the **training_metric.json** at the root directory.

# Requirment
Python 3.7

scikit-learn

torch 1.11.0

transformer 4.18.0


You may find other packages in the **requirement.txt** and **requirement_conda.txt** for pip and conda environment.
