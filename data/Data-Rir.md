# 📁 Dataset Directory Structure for Celeb-ReID

This follows the **same directory layout as Market1501** and supports RGB, IR, and TI modalities along with generated captions.

```bash
Data/
├── Celeb-reID/
│   ├── train/
│   │   ├── visible/*.jpg
│   │   ├── IR/*.jpg
│   │   └── TI/*.jpg
│   ├── query/                # same three sub-folders
│   ├── gallery/              # same three sub-folders
├── cap_predictions/          # ← captions root
│   └── Celebrity Captions/   # ← keep txt files here
│       ├── Celeb-bbox-train-visible.txt
│       ├── Celeb-bbox-train-IR.txt
│       └── ...
```

## 📝 Notes:
- **visible/** contains RGB training images.
- **IR/** and **TI/** contain infrared and thermal images respectively.
- The `cap_predictions/` folder holds caption files generated for each modality.
- TXT files follow naming conventions like: `Celeb-bbox-train-visible.txt`.

