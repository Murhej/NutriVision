# NutriVision Mobile App

React Native (Expo) mobile frontend for the NutriVision nutrition tracking system.

## Prerequisites

- **Node.js** v18 or higher — [download here](https://nodejs.org/)
- **Android Studio** — for running the app in the Android Emulator

## Setup

### 1. Install dependencies

```bash
cd mobile
npm install
```

### 2. Start the Expo development server

```bash
npx expo start
```

This will open the Expo Dev Tools in your terminal showing a QR code and options.

## Running on Android Emulator (Android Studio)

### Step 1: Set up an Android Virtual Device (AVD)

1. Open **Android Studio**
2. Go to **Tools → Device Manager** (or click the phone icon in the toolbar)
3. Click **Create Device**
4. Select a phone like **Pixel 7** → click **Next**
5. Choose a system image (download one if needed, e.g. **API 34**) → click **Next**
6. Click **Finish**

### Step 2: Launch the Emulator

1. In the Device Manager, click the **▶ Play** button next to your AVD
2. Wait for the emulator to fully boot up (you should see the Android home screen)

### Step 3: Run the app

Open a terminal in the `mobile/` directory and run:

```bash
npx expo start --android
```

Expo will automatically detect the running emulator and install the app. The NutriVision app will open inside the emulator.

### Alternative: Press `a` in the terminal

If you already ran `npx expo start`, simply press the **`a`** key in the terminal to launch on the Android emulator.

## Running on a Physical Phone

1. Install the **Expo Go** app from the Play Store / App Store
2. Run `npx expo start` in the `mobile/` directory
3. Scan the QR code shown in the terminal with your phone's camera
4. The app will open inside Expo Go

> **Note:** Your phone and computer must be on the same Wi-Fi network.

## Project Structure

```
mobile/
├── App.js                    # Entry point
├── src/
│   ├── theme/                # Design system
│   │   ├── colors.js         # Color tokens (light/dark)
│   │   ├── typography.js     # Typography, spacing, shadows
│   │   └── ThemeContext.js   # React context for theme toggle
│   ├── components/           # Reusable UI components
│   │   ├── Button.js
│   │   ├── Card.js
│   │   ├── ProgressRing.js
│   │   ├── ProgressBar.js
│   │   ├── IconBadge.js
│   │   └── MealCard.js
│   ├── screens/              # App screens
│   │   ├── WelcomeScreen.js
│   │   ├── DashboardScreen.js
│   │   ├── CalendarScreen.js
│   │   ├── ScanScreen.js
│   │   ├── FeedScreen.js
│   │   ├── ProfileScreen.js
│   │   └── LeaderboardScreen.js
│   ├── navigation/
│   │   └── AppNavigator.js   # Stack + Tab navigation
│   └── data/
│       └── mockData.js       # Sample data for all screens
├── package.json
└── app.json
```

## Features (Week 1 & 2)

- ✅ Premium light/dark mode design system
- ✅ Reusable component library (Button, Card, ProgressRing, etc.)
- ✅ Welcome/onboarding screen
- ✅ Home dashboard with calorie ring & macro tracking
- ✅ Calendar with color-coded daily tracking
- ✅ Food scan/analysis screen (mock AI predictions)
- ✅ Article feed with category filters
- ✅ Profile with achievements & settings
- ✅ Leaderboard with podium & rankings
- ✅ Bottom tab navigation with icons

## Troubleshooting

### Emulator not detected

Make sure the emulator is **fully booted** before running `npx expo start --android`. You can verify by checking that the Android home screen is visible.

### Metro bundler errors

Try clearing the cache:

```bash
npx expo start --clear
```

### Port conflict

If port 8081 is in use:

```bash
npx expo start --port 8082
```
