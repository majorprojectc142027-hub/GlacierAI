\# GlacierAI – AI-Based Global Glacier Monitoring and Retreat Analysis



GlacierAI is an AI-powered glacier monitoring system designed to analyze glacier boundaries, estimate glacier area, monitor changes over time, and support the study of glacier retreat using satellite imagery and geospatial data.



The project combines deep learning, satellite imagery, GIS processing, and an interactive web dashboard to support global glacier monitoring.



\## 1. Objectives



\- Detect glacier boundaries from satellite imagery using deep learning.

\- Identify and classify different glacier types.

\- Estimate glacier area using predicted segmentation masks.

\- Analyze historical glacier extent and retreat.

\- Identify potential glacier change hotspots.

\- Develop an interactive GIS-based monitoring dashboard.



\## 2. Key Features



\- \*\*Glacier Search:\*\* Search for glaciers using glacier names, RGI IDs, or geographical regions.

\- \*\*Glacier Mapping:\*\* Display glacier locations and boundaries using geospatial data.

\- \*\*AI Segmentation:\*\* Generate glacier masks from satellite imagery using a U-Net model.

\- \*\*Glacier Classification:\*\* Classify glacier scenes using a trained deep-learning model.

\- \*\*Area Analysis:\*\* Estimate glacier area from segmentation results.

\- \*\*Live Analysis:\*\* Retrieve satellite imagery and perform glacier analysis.

\- \*\*Debris Analysis:\*\* Analyze glacier debris-related information using imagery and segmentation results.

\- \*\*Historical Analysis:\*\* Support historical glacier comparisons using available observations and stored data.



\## 3. Technology Stack



| Component | Technologies |

|---|---|

| Frontend | React, TypeScript, Vite |

| Backend | Python, FastAPI |

| Deep Learning | PyTorch, Torchvision |

| Image Processing | OpenCV, Pillow, SciPy |

| Geospatial Processing | Rasterio, GeoPandas, Shapely, PyProj |

| Data Processing | NumPy, Pandas |

| Visualization | Leaflet, React Leaflet, Recharts |

| Satellite Data | Sentinel-2 |

| Glacier Inventory | Randolph Glacier Inventory (RGI) |



\## 4. Project Structure



```text

GlacierAI/

├── backend/

│   ├── app/

│   │   ├── api/

│   │   ├── services/

│   │   └── ai/

│   ├── models/

│   ├── scripts/

│   ├── checkpoints/

│   ├── requirements.txt

│   └── ...

├── frontend/

│   ├── public/

│   ├── src/

│   ├── package.json

│   └── ...

├── datasets/

│   ├── raw/

│   ├── processed/

│   ├── live/

│   ├── historical/

│   └── training\_v2/

├── checkpoints/

├── docs/

├── experiments/

├── .env.example

└── README.md

```



Large datasets, trained checkpoints, environment folders, and generated outputs are maintained separately from the source-code repository.



\## 5. Installation and Setup



\### Prerequisites



\- Python 3.11

\- Node.js and npm

\- Git



\### Clone the Repository



```bash

git clone https://github.com/majorprojectc142027-hub/GlacierAI.git

cd GlacierAI

```



\### Backend Setup



Create and activate a virtual environment:



```bash

cd backend

python -m venv venv

```



Windows PowerShell:



```powershell

.\\venv\\Scripts\\Activate.ps1

```



Install dependencies:



```bash

python -m pip install --upgrade pip

pip install -r requirements.txt

```



Install a compatible PyTorch and Torchvision build for your CPU or CUDA environment if they are not already installed.



\### Frontend Setup



Open another terminal:



```bash

cd frontend

npm install

```



\## 6. Environment Configuration



Create a `.env` file in the project root using `.env.example` as a template.



Configure the required Copernicus Data Space credentials:



```env

CDSE\_USERNAME=

CDSE\_PASSWORD=

CDSE\_CLIENT\_ID=

CDSE\_CLIENT\_SECRET=

```



Do not upload `.env` or expose API credentials publicly.



\## 7. Dataset and Model Configuration



GlacierAI uses satellite imagery, glacier inventory data, and trained model checkpoints.



Required runtime resources include:



\- Processed global RGI inventory: `datasets/processed/global\_rgi.gpkg`

\- U-Net segmentation checkpoint: `checkpoints/best\_unet\_v5.pth`

\- Glacier classification checkpoint: `backend/checkpoints/glacier\_classifier\_scene\_v1.pth`



Additional datasets are needed for training, historical analysis, and specific workflows.



Refer to `DATASET\_GUIDE.md` and `MODEL\_GUIDE.md` for download instructions and required folder locations.



\## 8. Running the Application



\### Start the Backend



From the project root:



```bash

python -m uvicorn backend.app.main:app --reload --app-dir .

```



The backend API will be available at:



```text

http://127.0.0.1:8000

```



\### Start the Frontend



From the frontend directory:



```bash

npm run dev

```



Open the local URL displayed by Vite in your browser.



\## 9. Future Enhancements



\- Improved global glacier segmentation accuracy.

\- More detailed debris-covered glacier classification.

\- Multi-temporal glacier retreat analysis.

\- Future glacier area prediction.

\- Explainable AI and model uncertainty visualization.

\- Expanded global glacier monitoring coverage.



\## 10. Project Scope



GlacierAI aims to provide an integrated platform for AI-based glacier boundary detection, glacier classification, area estimation, and retreat analysis using satellite imagery and geospatial datasets.



