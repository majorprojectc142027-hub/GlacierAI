\# GlacierAI – Dataset Guide



This guide explains the dataset folders used by GlacierAI and where the required data should be placed.



\## 1. Dataset Directory Structure



```text

datasets/

├── raw/

│   ├── RGI/

│   └── Sentinel2/

├── processed/

│   └── global\_rgi.gpkg

├── training\_v2/

├── live/

│   ├── large\_glacier/

│   ├── predictions/

│   └── sentinel2\_cache/

├── historical/

│   ├── predictions/

│   ├── satellite/

│   └── glacier\_data/

└── diagnostics/

```



\## 2. Dataset Description



| Dataset | Purpose |

|---|---|

| RGI | Glacier inventory and boundary information |

| Sentinel-2 | Satellite imagery for glacier analysis |

| Processed RGI | Prepared global glacier inventory |

| Training V2 | Prepared training images and masks |

| Live | Live analysis outputs and cached imagery |

| Historical | Historical glacier observations and predictions |

| Diagnostics | Diagnostic and experiment outputs |



\## 3. Required Dataset



\### Processed Global RGI Inventory



File:



```text

datasets/processed/global\_rgi.gpkg

```



This file is used by the glacier search and geospatial services.



\### Raw RGI Data



Location:



```text

datasets/raw/RGI/

```



Original RGI files are used by data preparation, processing, and training scripts.



\### Sentinel-2 Data



Location:



```text

datasets/raw/Sentinel2/

```



Used by local satellite-data workflows. Some live-analysis features also retrieve fresh satellite imagery through the configured data service.



\## 4. Training Data



Location:



```text

datasets/training\_v2/

```



Contains prepared glacier image patches and corresponding masks used by model-training scripts.



This dataset is required when training or reproducing the training workflow, but is not automatically required merely to load an existing trained model.



\## 5. Live and Historical Data



\- `datasets/live/` stores live-analysis outputs, prediction files, and satellite imagery caches.

\- `datasets/historical/` stores historical glacier data and previously generated analysis results.

\- `datasets/diagnostics/` stores diagnostic information.



These folders support their respective workflows. Their contents may be generated during application use or supplied separately.



\## 6. Important Notes



\- Large satellite datasets should be stored outside GitHub.

\- Preserve the directory structure when copying datasets into the project.

\- Do not rename files referenced by application scripts.

\- Do not upload private credentials or unnecessary generated files to GitHub.

\- Training datasets and runtime datasets have different requirements.



\## 7. Dataset Availability



Dataset download links and access instructions should be added here when the corresponding data packages are prepared for distribution.


## Downloading Datasets

All required datasets are available here:

[Download GlacierAI Datasets]([https://drive.google.com/drive/folders/14NJRWzdT7fXGDfUfTEfJZZRLLgEjzZB7?usp=sharing](https://drive.google.com/drive/folders/14NJRWzdT7fXGDfUfTEfJZZRLLgEjzZB7?usp=sharing))

Download the required files and maintain the folder structure described above.
