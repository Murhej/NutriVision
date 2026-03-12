# NutriVision

NutriVision is a food image classification and meal logging project built with PyTorch and FastAPI. It predicts foods from images, lets the user confirm or correct the result, estimates nutrition through external nutrition APIs, and supports incremental fine-tuning on top of an existing Food-101 model.

## What it does

- Classifies meal photos with top-3 predictions.
- Lets the user rename a meal or open a correction flow when the prediction is wrong.
- Collects 3 meal photos plus optional nutrition-label evidence for training review.
- Estimates nutrition through real providers instead of hardcoded calorie tables.
- Supports incremental training with extra datasets without restarting from scratch.
- Saves the best checkpoint and evaluation report for the active model.

## Main features

### Model and training

- Food-101 baseline training in `src/train_food101.py`
- Incremental fine-tuning in `src/incremental_train.py`
- Per-class Top-1 and Top-3 evaluation in `src/analyze_performance.py`
- Windows-aware CUDA safety paths for training and evaluation
- Top-1 and Top-3 reporting throughout training and evaluation

### Web app

- Upload a meal image or test with dataset images
- See top-3 predictions and confidence
- Use a quick meal rename when the prediction is close
- Open a correction page when the prediction is wrong
- Upload top, side, inside, and optional nutrition-facts images
- Adjust portion size dynamically in the frontend
- Save meal logs through the backend

### Nutrition lookup

- Tries Edamam first
- Tries USDA FoodData Central second
- Returns provider-aware errors when the issue is network, auth, or rate limiting
- Gives follow-up questions and stronger search queries when no nutrition match is found

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Optional: check GPU

```bash
python check_gpu.py
```

### 3. Train the Food-101 baseline

```bash
python -m src.train_food101
```

This writes:

- `runs/best_model.pth`
- `runs/report.json`

### 4. Optional: build per-class analysis

On Windows, CPU is the safer option for full-set evaluation:

```bash
python src/analyze_performance.py --device cpu
```

This writes:

- `outputs/per_class_performance.json`

### 5. Start the API

```bash
python -m src.api
```

Open:

- Web UI: `http://127.0.0.1:8000/static/index.html`
- API docs: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/health`

## Nutrition API setup

The app uses real nutrition providers. Do not commit keys to git.

Create a local file named `.env.local` in the project root:

```env
EDAMAM_APP_ID=your_edamam_app_id
EDAMAM_API_KEY=your_edamam_api_key
USDA_API_KEY=your_data_gov_key
```

Notes:

- `.env.local` is ignored by git.
- If Edamam fails, the app will try USDA FoodData Central.
- If both fail, the UI will show a provider error instead of pretending the food was not found.

## Recommended manual tests

### Basic nutrition lookup

1. Start the backend with `python -m src.api`
2. Open `http://127.0.0.1:8000/static/index.html`
3. Upload a food image
4. Try one description only in the meal description box

Examples:

- `1 cup chicken fried rice with egg`
- `1 bowl beef noodle soup`
- `1 fillet fried fish`
- `2 skewers chicken satay with peanut sauce`

### Correction flow

When a meal is not matched:

1. Click the correction page link
2. Upload:
   - top image
   - side image
   - inside image
   - optional nutrition label image
3. Fill in meal name, type, brand, portion, and nutrition details
4. Submit to save training feedback into the backend submission folder

## Incremental training

Use incremental training when you want to add new food, fruit, or vegetable classes on top of the current best Food-101 checkpoint.

### Run it

```bash
python -m src.incremental_train
```

### Optional dataset download helper

```bash
python -m src.download_kaggle_datasets
```

### What incremental training does

- Loads the current `runs/best_model.pth`
- Reads extra datasets under `data/`
- Expands the classifier for new classes
- Replays some Food-101 samples to reduce forgetting
- Fine-tunes and keeps the new best checkpoint only when it passes the configured thresholds

### Supported extra sources

The trainer can work with:

- raw known datasets under `data/`
- incremental folder layouts under `data/*_incremental/`
- discovered image-folder style datasets under `data/`

Examples already used in this repo:

- Fruits-360
- Food Recognition 2022
- Indonesian Food Dataset
- fast food classification datasets
- custom incremental datasets

## Commands

### Training

```bash
python -m src.train_food101
python -m src.incremental_train
```

### Evaluation

```bash
python src/analyze_performance.py --device cpu
```

### API

```bash
python -m src.api
```

### Syntax checks

```bash
python -m py_compile src/api.py src/api_mapper.py src/food_mapper.py src/feedback_api.py
python -m py_compile src/train_food101.py src/incremental_train.py src/analyze_performance.py
node --check static/app.js
node --check static/correction.js
```

## Project structure

```text
NutriVision/
|-- src/
|   |-- api.py
|   |-- api_mapper.py
|   |-- food_mapper.py
|   |-- feedback_api.py
|   |-- train_food101.py
|   |-- incremental_train.py
|   |-- analyze_performance.py
|   `-- download_kaggle_datasets.py
|-- static/
|   |-- index.html
|   |-- app.js
|   |-- correction.html
|   |-- correction.js
|   |-- help.html
|   `-- styles.css
|-- data/
|-- outputs/
|-- runs/
|-- requirements.txt
|-- .gitignore
`-- README.md
```

## Important generated files

- `runs/best_model.pth`: active best checkpoint
- `runs/report.json`: active model report and class list
- `runs/last_incremental_report.json`: saved when an incremental run is rejected
- `outputs/per_class_performance.json`: per-class evaluation report
- `outputs/training_submissions/`: correction samples and manifest

## Troubleshooting

### Nutrition lookup fails for every food

Check:

- your backend is running from `python -m src.api`
- you are opening `http://127.0.0.1:8000/static/index.html`
- your API keys are set in `.env.local`
- outbound HTTPS is not blocked by firewall, VPN, antivirus, or proxy

### Edamam rate limit reached

- use your own Edamam credentials
- wait for rate limits to reset
- keep USDA enabled as the second provider

### USDA or Edamam auth errors

Check:

- `EDAMAM_APP_ID`
- `EDAMAM_API_KEY`
- `USDA_API_KEY`

### Windows CUDA issues during evaluation

Use CPU for full-image evaluation:

```bash
python src/analyze_performance.py --device cpu
```

### Model not loading

Make sure these exist:

- `runs/best_model.pth`
- `runs/report.json`

If not, train first:

```bash
python -m src.train_food101
```

## Deployment notes

Do commit:

- `src/`
- `static/`
- `requirements.txt`
- `README.md`
- `.gitignore`

Do not commit:

- `.env`
- `.env.local`
- `data/`
- `runs/`
- `outputs/`
- `*.pth`
- `*.pt`
- `*.log`

For deployment secrets, configure environment variables in your hosting platform instead of storing them in git.

## References

- Food-101 dataset: https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/
- Edamam Nutrition Data API: https://developer.edamam.com/edamam-docs-nutrition-api
- USDA FoodData Central: https://fdc.nal.usda.gov/api-guide/
