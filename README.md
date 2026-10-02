# MIG-GT

Source code and dataset of the paper "[Modality-Independent Graph Neural Networks with Global Transformers for Multimodal Recommendation](https://arxiv.org/abs/2412.13994)", which is accepted by AAAI 2025.



## Homepage and Paper

+ Homepage (MIG-GT): https://github.com/CrawlScript/MIG-GT
+ Paper Access:
    - **ArXiv**: [https://arxiv.org/abs/2412.13994](https://arxiv.org/abs/2412.13994) 



## Dataset

The dataset is the same as the one used in the paper 'A Tale of Two Graphs: Freezing and Denoising Graph Structures for Multimodal Recommendation.' Please refer to their official repository to download the pre-processed dataset, which should be placed in the data directory.


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
python main.py --gpu 0 --seed 1 --dataset $DATASET --result_dir results --method mig_gt
# Note: $DATASET can be 'baby', 'sports', or 'clothing'.
```




__License:__ [GPLv3](https://www.gnu.org/licenses/gpl-3.0.en.html)

Copyright (c) 2023-2024 Xtra Computing Group, NUS, Singapore.

