# 📁 Dataset Directory Structure for Celeb-ReID

This follows the **same directory layout as Market1501** and supports RGB, IR, and TI modalities along with generated captions.

```bash
Data/
# 📁 Dataset Directory Structure for Celeb-ReID

This follows the **same directory layout as Market1501** and supports RGB, IR, and TI modalities along with generated captions.

```bash
Data/
├── Market1501/
        cap_predictions
            Market1501 Captions
                └── Celebrity Captions/   # ← keep txt files here
               ├── Celeb-bbox-train-visible.txt
               ├── Celeb-bbox-train-IR.txt
               └── ...
        Market1501
           ├── train/
        │   │   ├── visible/*.jpg
        │   │   ├── IR/*.jpg
        │   │   └── TI/*.jpg
        │   ├── query/                # same three sub-folders
        │   ├── gallery/              # same three sub-folders
         
    ```

## 📝 Notes:
- **visible/** contains RGB training images.
- **IR/** and **TI/** contain infrared and thermal images respectively.
- The `cap_predictions/` folder holds caption files generated for each modality.
- TXT files follow naming conventions like: `Celeb-bbox-train-visible.txt`.


```

## 📝 Notes:
- **visible/** contains RGB training images.
- **IR/** and **TI/** contain infrared and thermal images respectively.
- The `cap_predictions/` folder holds caption files generated for each modality.
- TXT files follow naming conventions like: `Celeb-bbox-train-visible.txt`.

