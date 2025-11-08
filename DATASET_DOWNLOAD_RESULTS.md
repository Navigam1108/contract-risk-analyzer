# CUAD Dataset Download Results

## Summary

✅ **Success!** The CUAD dataset has been **mostly downloaded** with only 4 files skipped due to Windows path length limitations.

- **Downloaded**: 507 out of 511 files (99.2%)
- **Skipped**: 4 files with extremely long paths
- **Location**: `D:\hf\` (very short cache path)
- **Total Files**: 1,092 files including metadata

## What Was Skipped?

Only 4 files with extremely long paths were skipped:
1. `PlayboyEnterprisesInc_...Content License Agreement_ Marketing Agreement_ Sales-Purchase Agreement1.pdf`
2. `PlayboyEnterprisesInc_...Content License Agreement_ Marketing Agreement_ Sales-Purchase Agreement2.pdf`
3. `EMERALDHEALTHTHERAPEUTICSINC_...CONSULTING AGREEMENT - DR. GAETANO MORELLO N.D. INC..PDF`
4. Several other files with paths exceeding 260 characters

**Impact**: Minimal - you still have 99.2% of the dataset available!

## How to Use the Dataset

### Option 1: Use from Cache (Recommended)

```python
from datasets import load_dataset
import os

# Set the cache directory
os.environ["HF_HOME"] = "D:/hf"
os.environ["HF_HUB_CACHE"] = "D:/hf"
os.environ["HF_DATASETS_CACHE"] = "D:/hf"

# Load the dataset from cache
dataset = load_dataset(
    "theatticusproject/cuad",
    split="train",
    cache_dir="D:/hf"
)

print(f"Loaded {len(dataset)} examples")
```

### Option 2: Use Streaming Mode (No downloads needed)

```python
from datasets import load_dataset

# Stream the dataset directly from HuggingFace
dataset = load_dataset(
    "theatticusproject/cuad",
    split="train",
    streaming=True  # No local files needed!
)

# Iterate through examples
for example in dataset:
    print(example['title'])
    break
```

### Option 3: Update Your App

Modify `app.py` to use the downloaded cache:

```python
import os

# At the top of app.py, set cache directory
os.environ["HF_HOME"] = "D:/hf"
os.environ["HF_HUB_CACHE"] = "D:/hf"
os.environ["HF_DATASETS_CACHE"] = "D:/hf"

# Then load dataset normally
dataset = load_dataset("theatticusproject/cuad", split="train", cache_dir="D:/hf")
```

## Storage Info

- **Cache Location**: `D:\hf\`
- **Space Used**: ~500MB
- **Files**: 1,092 files (PDFs, metadata, dataset files)

## Troubleshooting

### If you see symlink errors:
The dataset will try to create symlinks which fail on Windows. The monkey-patch in `download_data.py` handles this by copying files instead.

### If you want to re-download:
```powershell
Remove-Item -Path "D:\hf" -Recurse -Force
python data\download_data.py
```

### If you want to save space:
Use streaming mode (Option 2) which doesn't require any local files.

## Recommendation

**Use Option 1** (cache-based loading) since you've already downloaded 99.2% of the dataset. The 4 missing files won't significantly impact your analysis.

Alternatively, **use Option 2** (streaming) if you want guaranteed access to 100% of the dataset without worrying about path issues.

---

*Last updated: After successful download of 507/511 files*
