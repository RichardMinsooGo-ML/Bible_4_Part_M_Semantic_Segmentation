# FCN

## Engilish
*  **Theory** : [https://wikidocs.net/227364](https://wikidocs.net/227364) <br>
*  **Implementation** : [https://wikidocs.net/227365](https://wikidocs.net/227365)

## 한글
*  **Theory** : [https://wikidocs.net/225904](https://wikidocs.net/225904) <br>
*  **Implementation** : [https://wikidocs.net/225991](https://wikidocs.net/225991)

## Step 1. Clone from Github and install library

Git clone to root directory. 

```Shell
# Clone from Github Repository
! git init .
! git remote add origin https://github.com/RichardMinsooGo-ML/Bible_4_Part_M_Semantic_Segmentation.git
# ! git pull origin master
! git pull origin main
```

A tool to count the FLOPs of PyTorch model.

```
from IPython.display import clear_output
clear_output()
```

## Training test
#### Download VOC Dataset

```Shell
# VOC 2012 Dataset Download and extract

! wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
!tar -xvf "/content/VOCtrainval_11-May-2012.tar" -C "/content/dataset"
clear_output()

# VOC 2007 Dataset Download and extract

! wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
! wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
!tar -xvf "/content/VOCtrainval_06-Nov-2007.tar" -C "/content/dataset"
!tar -xvf "/content/VOCtest_06-Nov-2007.tar" -C "/content/dataset"
clear_output()
```

#### Train

```
! python fcn.py
```

# U-net


## Engilish
*  **Theory** : [https://wikidocs.net/227366](https://wikidocs.net/227366) <br>
*  **Implementation** : [https://wikidocs.net/227367](https://wikidocs.net/227367)

## 한글
*  **Theory** : [https://wikidocs.net/225905](https://wikidocs.net/225905) <br>
*  **Implementation** : [https://wikidocs.net/225992](https://wikidocs.net/225992)

## Step 1. Clone from Github and install library

Git clone to root directory. 

```Shell
# Clone from Github Repository
! git init .
! git remote add origin https://github.com/RichardMinsooGo-ML/Bible_4_Part_M_Semantic_Segmentation.git
# ! git pull origin master
! git pull origin main
```

A tool to count the FLOPs of PyTorch model.

```
from IPython.display import clear_output
clear_output()
```

## Training test
#### Download VOC Dataset

```Shell
# VOC 2012 Dataset Download and extract

! wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
!tar -xvf "/content/VOCtrainval_11-May-2012.tar" -C "/content/dataset"
clear_output()

# VOC 2007 Dataset Download and extract

! wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
! wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
!tar -xvf "/content/VOCtrainval_06-Nov-2007.tar" -C "/content/dataset"
!tar -xvf "/content/VOCtest_06-Nov-2007.tar" -C "/content/dataset"
clear_output()
```

#### Train

```
! python u_net.py
```

