# Halu caro

### Pytorch (python 3.6 and up)
 
- Install `https://pytorch.org/get-started/locally/`
 
- Sửa thiết lập trong const.py

- `python train.py`

### Keras (python <=3.6)

- `pip install keras`

### Conda thiết lập python 3.6

- Tạo env mới `conda create -n py3.68 python=3.6.8`

- Switch env `conda activate py3.68`

### Google colab python 3 GPU

```
!pip install keras
!mkdir projects
!cd projects && mkdir caro
```
```
!cd projects/caro && python train.py
```

!mkdir projects

### Auto start training

- Open file `halucaro-ai-train.bat`

- Edit project folder path at line 3

- Edit Anaconda activate script path at line 4

- Create a shortcut file of `halucaro-ai-train.bat` and move it to `shell:startup`

### Auto backup script

- Open file `halucaro-models-backup.bat`

- Edit `backup.exe` file path at line 3

- Edit models folder path at line 3

- Create a shortcut file of `halucaro-models-backup.bat` and move it to `shell:startup`

### Play game

- `python human_play.py`

### enjoy :)