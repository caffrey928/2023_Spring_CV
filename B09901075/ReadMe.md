# (2023 Spring) CV Final

[Part 1: Prepare DataSet & Enviornment](#Prepare-Dataset-amp-Environment)

[Part 2: Train Yolo Model](#Train-Yolo-Model)

[Part 3: Evaluation](#Evaluation)

[Part 4: Generate Solution](#Generate-Solution)

[Summary](#Summary)
- Concludes all commands you need to run to generate result
- Show the whole folder structure

[Other](#Other)
- Not related to the best solution
- Intro about **traditional_cv/**

### Reminder: 
- The commands below use "python3" to run python scripts.
Remember to modify command if your environment can only use "python".
(In "Prepare Dataset" part, you may also have to change "python3" or "pip" in the bash script)

- About the example command in this ReadMe, **data/** is the folder for original data provided by TAs ; **datasets/** is the special dataset generated from **data/** for training yolo model. 


## Prepare Dataset & Environment

### Step 1:
- Download S1-S8 provided by TAs. 
- Put them into a folder. 
- These data are mainly for evaluation and creating solution.

#### Example Structure
```
-- data
     |--- S1
     |--- S2
     |--- S3
     ...
```

### Step 2:

<p>
In this step, because we need to train yolo model, so we have to create dataset with specific format from original dataset. 
</p>
<p>
For convenience, we provide the script to download the zip file of specific dataset, and also provide google drive link where we save dataset files and related code.
</p>

- To download specific dataset and install packages, run
```
bash download.sh
```
This scripts would first install packages (pip install), then download datasets.zip, unzip, and remove(only .zip file) it.

- If only wants to install packages, run
```
pip install -r requirements.txt
```

####  Intro about the provided Google Drive Link 
You don't have to see this part if you are not interested in the code about generating the special dataset.

[Google Drive Link](https://drive.google.com/drive/folders/1vEpwwSmDrbqVo1LUYA_AN8KQeHEmO8YA)

- This google drive space is for yolo method, the ReadME here explains the whole process about training yolo model.
- The code and description about train and predict are covered in below parts, so you don't really need to see these in google drive.
##### Reason we provide the link
- You can directly find datasets.zip. 
- Related code is in "data preprocessing/data_preprocessing.ipynb". 

## Train Yolo Model
- This part shows the code to get trained model. 
- We also provide a best.pt in our code submission, this is the model that gets the best score on codalab. You can just use this model if the train process is too long to run.
- To get a trained yolo model, run
```
python3 train_yolo.py [path to special dataset] [train_device]
```
- [train_device] can be either "0" (use gpu) or "cpu"

After training, you can get a model called 'best.pt'.

#### Example Command & Folder Structure
Command:
```
python3 train_yolo.py ./datasets cpu
```
Structure:
```
-- datasets
     |--- test
     |--- train
     |--- valid
-- ReadMe.md
-- train_yolo.py
-- download.sh
...
```

## Evaluation
- To evaluate model performance on S1-S4, run
```
python3 eval.py [dataset_path] [model_path]
```
#### Example Command & Folder Structure
Command:
```
python3 eval.py data best.pt
```
Structure:
```
-- data
     |--- S1
     |--- S2
     |--- S3
     ...
-- ReadMe.md
-- eval.py
-- best.pt
...
```

## Generate Solution
- To generate predict result on S5-S8, run
```
python3 solution.py [dataset_path] [model_path] [solution_path]
```
#### Example Command & Folder Structure
Command:
```
python3 solution.py data best.pt solution
```
Structure:
```
-- data
     |--- S1
     |--- S2
     |--- S3
     ...
-- ReadMe.md
-- solution.py
-- best.pt
-- solution
     |--- S5
     |--- S6
     |--- S7
     |--- S8
...
```

## Summary
**Under B09901075/**

**1. Prepare Dataset & Environment**
```
bash download.sh
```
or if you only want to install packages (and download datasets.zip manually)
```
pip install -r requirements.txt
```
**2. Train Yolo Model (can be skipped and use best.pt we provide)**
```
python3 train_yolo.py ./datasets cpu
```
(cpu change to 0 would be faster if you have GPU)

**3. Evaluation (also can be skipped if you only interested in solution result)**
```
python3 eval.py data best.pt
```
**4. Generate Solution**
```
python3 solution.py data best.pt solution
```

**Folder Structure**
```
-- data
     |--- S1
     |--- S2
     |--- S3
     ...
-- datasets
     |--- test
     |--- train
     |--- valid
-- ReadMe.md
-- download.sh
-- train_yolo.py
-- eval.py
-- solution.py
-- best.pt
-- solution
     |--- S5
     |--- S6
     |--- S7
     |--- S8
...
```

## Other

**This part is related to the best result on codalab.**

In this project, the best score we get is from yolo model. Other than that, we also tried some classical CV ways to optimize our result. 

Though the score got lower, we still submits related code to prove our work.

### refine_pupil.py
**Usage**
- After output generated from yolo model or other algorithm, apply this function to output.
```
output, conf = yolo_algorithm(model , input_image)
output = refine_pupil(output)
```

### traditional_cv.py
**Usage**
- Contains my_awesome_algorithm() to generate mask and confidence. (same effect as yolo method)
```
output, conf = my_awesome_algorithm(input_image)
```