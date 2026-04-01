# NutriVision

NutriVision is an AI-powered food image classification and nutrition tracking system. It combines a PyTorch deep learning model with a FastAPI backend and a React Native (Expo) mobile app to let users scan meals, view nutrition data, track daily macros, and manage personal dietary goals.

## What it does

- Classifies meal photos using a trained Food-101 deep learning model with top-3 predictions
- Estimates nutrition (calories, protein, carbs, fat) through real APIs (Edamam, USDA)
- Tracks daily meals, calories, and macros on a personal dashboard
- Supports user authentication — each person gets their own isolated profile and meal history
- Allows users to customize daily calorie and macro goals from their profile
- Supports incremental fine-tuning to add new food classes without retraining from scratch

---

## 🚀 Full Setup Guide (For Teammates)

Follow these steps in order to get the entire app running on your machine.

### Prerequisites

Make sure you have the following installed:

| Tool | Version | Download |
|------|---------|----------|
| **Python** | 3.9+ | https://www.python.org/downloads/ |
| **Node.js** | 18+ | https://nodejs.org/ |
| **Expo Go** (on your phone) | Latest | App Store / Play Store |
| **Git** | Any | https://git-scm.com/ |

### Step 1: Clone the Repository

```bash
git clone https://github.com/Murhej/NutriVision.git
cd NutriVision
```

### Step 2: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Train the AI Model

> ⚠️ This step is required before the backend can classify food images. Training takes a few minutes in fast mode.

```bash
python main.py train
```

This creates:
- `runs/best_model.pth` — the trained model weights
- `runs/report.json` — model class list and metrics

### Step 4: Set Up Nutrition API Keys

Create a file named `.env.local` in the project root (this file is git-ignored):

```env
EDAMAM_APP_ID=your_edamam_app_id
EDAMAM_API_KEY=your_edamam_api_key
USDA_API_KEY=your_data_gov_key
```

- Get Edamam keys at: https://developer.edamam.com/
- Get a USDA key at: https://fdc.nal.usda.gov/api-guide/

### Step 5: Start the Backend Server

Open **Terminal 1** (PowerShell):

```bash
cd "path\to\NutriVision"
python main.py serve
```

You should see:
```
[OK] API ready!
INFO: Uvicorn running on http://0.0.0.0:8000
```

> ⚠️ **Keep this terminal open!** The mobile app needs the backend running to work.

### Step 6: Install Mobile App Dependencies

Open **Terminal 2** (PowerShell):

```bash
cd "path\to\NutriVision\mobile"
npm install
```

### Step 7: Start the Expo Dev Server

In the same Terminal 2:

```bash
npx expo start
```

### Step 8: Open on Your Phone

1. Open the **Expo Go** app on your phone
2. Scan the QR code shown in Terminal 2
3. The app will load on your device

> ⚠️ Your phone and your computer **must be on the same Wi-Fi network**.

### Step 9: Create Your Account

When the app loads, you'll see the Welcome screen:
1. Tap **"Create Account"**
2. Enter your name, email, and a password
3. You're in! Your personal dashboard is ready

Each teammate creates their own account. All meal logs and goals are isolated per user.

---

## 📱 Mobile App Features

| Feature | Description |
|---------|-------------|
| **Dashboard** | Daily calorie ring, macro progress bars, today's meal list |
| **Scan** | Take a photo or pick from gallery → AI classifies it → shows nutrition |
| **Calendar** | View past days' calorie totals at a glance |
| **Leaderboard** | See who's logging the most consistently |
| **Profile** | Edit name, goal, calorie/macro targets, toggle units, dark mode |
| **Auth** | Real login/register system with isolated user data |

---

## 🔧 Common Commands Reference

### Backend

```bash
# Train the model
python main.py train

# Start the API server
python main.py serve

# Run evaluation
python main.py evaluate
```

### Mobile App

```bash
# Install dependencies (first time only)
cd mobile
npm install

# Start Expo dev server
npx expo start

# Clear cache and restart (if you see stale data)
npx expo start -c
```

---

## Project Structure

```text
NutriVision/
├── main.py                    # CLI entry point (train, serve, evaluate)
├── requirements.txt           # Python dependencies
├── .env.local                 # API keys (git-ignored, create manually)
│
├── src/
│   ├── api/
│   │   ├── app.py             # FastAPI application factory
│   │   ├── auth.py            # User authentication (register/login)
│   │   ├── inference.py       # Image classification endpoint
│   │   ├── nutrition.py       # Nutrition lookup & meal logging
│   │   ├── mobile_sync.py     # Dashboard, calendar, profile, leaderboard APIs
│   │   ├── food_mapper.py     # Edamam & USDA nutrition query engine
│   │   └── feedback.py        # Correction flow API
│   ├── training/              # Model training scripts and config
│   ├── core/                  # Data loading, model registry
│   ├── evaluation/            # Per-class accuracy analysis
│   └── visualization/         # Training charts and plots
│
├── mobile/
│   ├── App.js                 # React Native entry point
│   ├── package.json           # Node.js dependencies
│   └── src/
│       ├── api/client.js      # API client with auth token management
│       ├── context/AuthContext.js  # Authentication state provider
│       ├── navigation/AppNavigator.js  # Route definitions
│       ├── screens/
│       │   ├── WelcomeScreen.js
│       │   ├── LoginScreen.js
│       │   ├── RegisterScreen.js
│       │   ├── DashboardScreen.js
│       │   ├── ScanScreen.js
│       │   ├── CalendarScreen.js
│       │   ├── ProfileScreen.js
│       │   ├── EditProfileScreen.js
│       │   └── LeaderboardScreen.js
│       ├── components/        # Reusable UI components
│       └── theme/             # Design tokens and dark mode
│
├── static/                    # Legacy web UI
├── outputs/                   # Generated data (meal_logs.json, users.json)
├── runs/                      # Model checkpoints and reports
└── data/                      # Training datasets
```

---

## Troubleshooting

### "Connection Refused" on the mobile app

- Make sure `python main.py serve` is running in a separate terminal
- Make sure your phone and computer are on the **same Wi-Fi network**
- Try restarting Expo with cache clear: `npx expo start -c`

### Model not loading when starting the server

```bash
python main.py train
```

### Nutrition lookup returns nothing

- Check that `.env.local` has valid API keys
- Try searching for simpler food names (e.g., "pizza" instead of "pepperoni pizza with extra cheese")
- Make sure outbound HTTPS is not blocked by firewall, VPN, antivirus, or proxy

### Nutrition lookup fails for every food

Check:

- your backend is running from `python main.py serve`
- your API keys are set in `.env.local`
- outbound HTTPS is not blocked by firewall, VPN, antivirus, or proxy

### Edamam rate limit reached

- Use your own Edamam credentials
- Wait for rate limits to reset
- Keep USDA enabled as the second provider

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

### App shows stale/old data

- Press `r` in the Expo terminal to reload the app
- Or restart with `npx expo start -c`

### AsyncStorage error

Make sure you installed the Expo-compatible version:
```bash
cd mobile
npx expo install @react-native-async-storage/async-storage
```

---

## Optional: Check GPU

```bash
python check_gpu.py
```

## Optional: Per-Class Analysis

On Windows, CPU is the safer option for full-set evaluation:

```bash
python src/analyze_performance.py --device cpu
```

This writes:

- `outputs/per_class_performance.json`

---

## Recommended Manual Tests

### Basic nutrition lookup

1. Start the backend with `python main.py serve`
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

---

## Incremental Training

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

---

## Important Generated Files

- `runs/best_model.pth`: active best checkpoint
- `runs/report.json`: active model report and class list
- `runs/last_incremental_report.json`: saved when an incremental run is rejected
- `outputs/per_class_performance.json`: per-class evaluation report
- `outputs/training_submissions/`: correction samples and manifest
- `outputs/users.json`: registered user accounts (git-ignored)
- `outputs/meal_logs.json`: all logged meals (git-ignored)

---

## Deployment Notes

Do commit:

- `src/`
- `mobile/`
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
- `node_modules/`

For deployment secrets, configure environment variables in your hosting platform instead of storing them in git.

---

## Syntax Checks

```bash
python -m py_compile src/api/app.py src/api/nutrition.py src/api/food_mapper.py src/api/feedback.py src/api/auth.py
node --check static/app.js
node --check static/correction.js
```

---

## References

- Food-101 dataset: https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/
- Edamam Nutrition Data API: https://developer.edamam.com/edamam-docs-nutrition-api
- USDA FoodData Central: https://fdc.nal.usda.gov/api-guide/
- Expo Documentation: https://docs.expo.dev/
- React Navigation: https://reactnavigation.org/

