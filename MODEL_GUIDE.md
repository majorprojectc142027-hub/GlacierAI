\# GlacierAI – Model Guide



This document describes the trained deep-learning models used by GlacierAI and their required locations.



\## 1. Model Overview



GlacierAI uses two primary trained models for its current application workflows.



| Model | Architecture | Purpose |

|---|---|---|

| Glacier Segmentation | U-Net | Detect glacier regions in satellite imagery |

| Glacier Classification | ResNet18 | Classify glacier scenes into three categories |



\## 2. U-Net Segmentation Model



\*\*Checkpoint:\*\*



```text

checkpoints/best\_unet\_v5.pth

```



\*\*Purpose:\*\*



\- Predict glacier regions from satellite imagery.

\- Generate binary glacier segmentation masks.

\- Support glacier boundary detection and area estimation.



\*\*Configuration from the application:\*\*



\- Model: Custom U-Net

\- Input size: 256 × 256

\- Input channels: 3

\- Output classes: 1

\- Prediction threshold: 0.5



The backend loads this checkpoint through:



```text

backend/app/ai/segmenter.py

```



\## 3. Glacier Classification Model



\*\*Checkpoint:\*\*



```text

backend/checkpoints/glacier\_classifier\_scene\_v1.pth

```



\*\*Architecture:\*\* ResNet18



\*\*Purpose:\*\*



Classifies glacier scenes into three categories:



1\. Glacieret / Snowfield

2\. Ice Cap

3\. Mountain Glacier



The model is loaded through:



```text

backend/services/glacier\_classifier\_service.py

```



\## 4. Model Directory Structure



```text

GlacierAI/

├── checkpoints/

│   └── best\_unet\_v5.pth

└── backend/

&#x20;   └── checkpoints/

&#x20;       └── glacier\_classifier\_scene\_v1.pth

```



Both checkpoint files must be placed in their exact locations for the corresponding application services to load them.



\## 5. Additional Experimental Checkpoints



The project also contains other checkpoints used for experiments, training, and evaluation.



Examples include:



\- `best\_unet.pth`

\- `best\_unet\_v2.pth`

\- `best\_unet\_v3.pth`

\- `best\_unet\_rgi\_v1.pth`

\- `glacier\_classifier\_v3.pth`

\- `glacier\_classifier\_scene\_v2.pth`

\- `glacier\_classifier\_context\_v2.pth`



These are not substitutes for the two runtime checkpoints unless the corresponding application code is updated and tested.



\## 6. Model Download and Setup



Trained model files are stored separately from the GitHub source repository because of their file sizes.



After obtaining the required checkpoints, place them in the directory structure shown above.



Do not rename the checkpoint files unless their paths are also updated in the backend code.



\## 7. Training



The project includes scripts under:



```text

backend/scripts/

```



Training workflows require the corresponding training datasets and compatible Python dependencies.



The existing trained checkpoints allow the application to perform inference without retraining the models.



