# MMTTA: Multimodality Test-Time Training Adaption

This repository implements **MMTTA**, our framework for **Multi-Modal Person Re-Identification**, extending multi-modality training by adapting the model at inference using relationships across RGB, IR, Thermal, and caption modalities.

---

## Features

- **Distillation L2 Loss**: Aligns student and teacher feature representations via normalized L2 distance:

  $$
  L_{distill} = \| \frac{f_{stu}}{\|f_{stu}\|_2} - \frac{f_{tea}}{\|f_{tea}\|_2} \|_2^2
  $$

  Implements simple distillation between ViT (student) and nomic (teacher) features. fileciteturn0file0

- **Multi-Modal Margin Loss**: Enforces a margin between modality-specific identity centers by penalizing the worst-case pair:

  For modalities \$i,j\in{RGB,IR,TI}\$ with centers \$c\_i, c\_j\$ and margin \$m\$:

  $$
  L_{margin} = \frac{1}{N}\sum_{k=1}^N \max_{\{i,j\}} \bigl| m - d(c_i^k, c_j^k) \bigr|
  $$

  where \$d(\cdot,\cdot)\$ is L2 distance and \$N\$ is number of identities in batch. fileciteturn0file1

- **Vision–Language InfoNCE Loss**: Aligns image and text features using a temperature-scaled InfoNCE objective:

  $$
  L_{InfoNCE} = -\frac{1}{B} \sum_{i=1}^B \log \frac{\exp(\cos(f_v^i,f_t^i)/\tau)}{\sum_{j=1}^B \exp(\cos(f_v^i,f_t^j)/\tau)}
  $$

  where \$B\$ is batch size and \$\tau\$ is temperature. fileciteturn0file2

---

## Environment & Dependencies

1. Clone the repository and install dependencies:
   ```bash
   git clone <this-repo-url>
   cd <repo-folder>
   pip install -r requirements.txt
   ```
2. Based on prior works:
   - TransReID (ICCV 2021)
   - IEEE AAAI 2022 multi-modal ReID

---

## Datasets

Prepare each person ReID dataset with RGB, IR, Thermal, and caption modalities:

- **Market1501** (and Market1501-MM)
- **Real2**
- **PRCC**
- **CUHK03**
- **MSMT17**

Organize under `data/{DatasetName}`:

```
data/
  Market1501/
    RGB/
    IR/
    Thermal/
    captions.txt
  PRCC/
    ...
```

Adjust dataset paths in `configs/{DatasetName}/` as needed.

---

## Training

Standard multi-modal training:

```bash
# Example: train on Market1501
python train.py --config_file configs/Market1501/vit_base.yml
```

Available configs:

- `configs/Market1501-MM/vit_base.yml`
- `configs/Real2/vit_base.yml`
- `configs/PRCC/vit_base.yml`
- `configs/CUHK03/vit_base.yml`
- `configs/MSMT17/vit_base.yml`

---

## Evaluation

After training epochs, metrics (mAP, CMC) are logged. To evaluate a saved checkpoint:

```bash
python test.py --config_file configs/Market1501/vit_base.yml \
               --model_path /path/to/checkpoint.pth
```

---

*This README focuses on code usage, dataset setup, and loss function definitions.*

