# 1-GAME
## Requirements

+ Linux
+ Python 3.7
+ torch==1.12.1+cu113
+ torchmetrics==0.11.4
+ dgl==1.0.2+cu113
+ ogb==1.3.5
+ shortuuid==1.0.11
+ pandas==1.3.5
+ numpy==1.21.6
+ tqdm==4.64.1



## RUN

```bash
# Run the following command:
python main.py --gpu 0 --seed 1 --dataset $DATASET --result_dir results --method one_game
# Note: $DATASET can be 'baby', 'sports', or 'clothing'.
```

