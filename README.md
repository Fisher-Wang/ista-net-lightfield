# ISTA-Net for lightfield

- `ista_net_plus_36.py`: network structure
- `Train_CS_ISTA_Net_plus_36.py`: training script
- `TEST_CS_ISTA_Net_plus_36_rgb.py`: testing script

To test the network, run: 
```python
python TEST_CS_ISTA_Net_plus_36_rgb.py --matrix_dir lf/myphi --model_dir lf/model --data_dir lf/mytrain --log_dir lf/log --layer_num 9 --data_dir lf/mytest --result_dir lf/result --cs_ratio 25
```
you can set `cs_ratio` to 4, 25, 40, 50

To train the network, run: 
```python
python Train_CS_ISTA_Net_plus_36.py --matrix_dir lf/myphi --model_dir lf/model --data_dir lf/mytrain --log_dir lf/log --data_name data4499 --cs_ratio 25
```

Requirements:
- GPU > 10GB

Environment:
- Pytorch 1.8

Reference:
- modified from [ISTA-Net](https://github.com/jianzhangcs/ISTA-Net-PyTorch)
