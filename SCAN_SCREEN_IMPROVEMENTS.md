# Scan Screen Improvements - Complete Documentation

## Overview

The Scan Screen has been completely rebuilt to provide an enhanced user experience with:
- **Proper Safe Area Handling**: Content is now properly positioned with safe area padding at top/bottom
- **Real Food Image Upload**: Full camera and gallery integration with actual image preview
- **External API Nutrition Integration**: Uses Edamam/USDA APIs for real nutrition facts
- **Portion Adjustment**: Dynamic scaling of nutrition values when portion size changes
- **Full Nutrition Facts Modal**: Comprehensive nutrition label display (8+ nutrients)
- **Error Handling**: Graceful fallback messages and timeout protection
- **Backend Persistence**: Meals are saved to backend and sync across app

---

## Key Features

### 1. Layout & Safe Area Fixes ✓

**Upload Step:**
- Camera icon in large circle (40pt icon, 80pt container)
- Title and subtitle text centered
- Take Photo and Upload buttons side-by-side
- Tips card with 4 best-practice items
- Proper top/bottom safe area padding with `useSafeAreaInsets()`
- Content pushed down from status bar
- Tab bar remains visible at bottom

**Portions Step:**
- Food image preview (240pt height)
- Confidence badge (top-right)
- Prediction selection cards
- Portion size buttons (Small/Medium/Large/Extra Large)
- Custom multiplier controls (+/- buttons)
- Nutrition breakdown grid (4 macros)
- "View Full Nutrition Facts" button

**Review Step:**
- Food image preview (200pt)
- Meal name and portion info
- Summary nutrition grid
- Full nutrition label button
- Back & Add to Log buttons

### 2. Image Handling ✓

**Camera:**
- Requests camera permissions
- Launches camera with editing enabled
- Quality: 80% (good balance of quality/file size)

**Upload:**
- Requests media library permissions
- Launches image gallery
- Supports common formats (JPG, PNG, GIF, WebP)
- Auto-detects MIME type based on file extension

**Processing:**
- Images sent to `/predict` endpoint for AI analysis
- Returns top 3 predictions with confidence scores
- Selected prediction used for nutrition lookup

### 3. Nutrition API Integration ✓

**Prediction Flow:**
```
1. Image upload → `/predict` endpoint
2. Returns: [
     { class: "grilled_salmon", confidence: 0.97 },
     { class: "baked_fish", confidence: 0.87 },
     { class: "cooked_fish", confidence: 0.75 }
   ]
3. Format labels: Replace underscores, show confidence %
```

**Nutrition Lookup:**
```
1. Selected prediction → `/map/nutrition` endpoint
2. Query: "grilled salmon" (cleaned class name)
3. Returns: {
     calories: 280,
     protein_g: 39.2,
     carbs_g: 0,
     fat_g: 12.8,
     fiber_g: 0,
     sugar_g: 0,
     sodium_mg: 92,
     cholesterol_mg: 68
   }
```

**Portion Scaling:**
```
1. Base nutrition × portion multiplier
2. Formula: nutrition_field * multiplier (0.25 to 4.0)
3. Multipliers:
   - Small: 0.68x
   - Medium: 1.0x
   - Large: 1.32x
   - Extra Large: 1.68x
   - Custom: +/- 0.25 increment buttons
```

### 4. Portion Adjustment ✓

**Preset Selection:**
- 4 buttons (Small, Medium, Large, Extra Large)
- Each shows portion weight (115g, 170g, 225g, 285g)
- Pressing a button sets custom multiplier to preset value

**Custom Multiplier:**
- Minus button: -0.25x
- Display: Current multiplier (e.g., "1.00x")
- Plus button: +0.25x
- Range: 0.25x to 4.0x
- Clamped and validated on save

**Dynamic Updates:**
- Watching: `nutrition`, `selectedPortionId`, `customMultiplier`
- Recalculates `scaledNutrition` on any change
- UI updates in real-time

### 5. Full Nutrition Facts Modal ✓

**Trigger:** "View Full Nutrition Facts" button on both portions & review steps

**Content:**
- Modal overlay with half-slide animation
- Header with title and X close button
- Subtitle with meal name & portion size
- Scrollable nutrition facts list:
  - Calories (kcal)
  - Protein (g)
  - Carbohydrates (g)
  - Dietary Fiber (g)
  - Sugars (g)
  - Total Fat (g)
  - Sodium (mg)
  - Cholesterol (mg)
- Disclaimer: "Values are estimated using AI image analysis and nutrition APIs..."
- Close button at bottom

**Styling:**
- Dark modal overlay
- Card-style sheet with rounded top corners
- Sticky header with border
- Values displayed in primary color

### 6. Error Handling ✓

**Image Processing Errors:**
- "Could not identify food in this image. Try another photo."
- User returned to upload step to retry

**Nutrition Fetch Errors:**
- "No nutrition data found for this food. Please select another option or adjust the meal name."
- User can select different prediction and retry

**Network/Timeout Errors:**
- 8-second timeout on regular API calls
- 15-second timeout on file uploads
- "Request timeout... Check your internet connection or server availability."

**Save/Persistence Errors:**
- "Could not save meal. Please check your internet connection and try again."
- Error box shown in red with error context
- User can retry or go back

### 7. Backend Integration ✓

**Save Endpoint:** `POST /map/log`

**Payload:**
```json
{
  "food_label": "grilled_salmon",
  "display_name": "Grilled Salmon",
  "portion_id": "medium",
  "portion_multiplier": 1.0,
  "nutrition": {
    "calories": 280,
    "protein_g": 39.2,
    "carbs_g": 0,
    "fat_g": 12.8,
    "fiber_g": 0,
    "sugar_g": 0,
    "sodium_mg": 92,
    "cholesterol_mg": 68
  },
  "prediction": {
    "label": "Grilled Salmon",
    "rawClass": "grilled_salmon",
    "confidence": 97,
    "emoji": "🍽️"
  },
  "source": "mobile_camera",
  "image_url": "file:///path/to/image.jpg"
}
```

**Response:**
```json
{
  "status": "saved",
  "entry": {
    "timestamp": "2024-04-16T12:30:45.123Z",
    "user_id": "user_123",
    ... (copy of payload)
  }
}
```

**Sync to App:**
- After successful save, user sees "Meal saved to your log!"
- "View Log" button navigates to Home
- Home/Dashboard fetches updated meals from `/api/mobile/dashboard`
- Calendar fetches from `/api/mobile/calendar`

### 8. State Management ✓

**Step States:**
- `'upload'`: Initial landing page (camera/upload buttons)
- `'portions'`: After image processed (predictions, nutrition, portion adjustment)
- `'review'`: Preview before saving (confirm details, save button)

**Data States:**
```javascript
imageUri          // Selected image file URI
predictions       // Array of food predictions with confidence
selectedPrediction // Index of selected prediction
nutrition         // Base nutrition data from API
scaledNutrition   // Nutrition scaled by portion multiplier
selectedPortionId // 'small' | 'medium' | 'large' | 'extra_large'
customMultiplier  // 0.25 to 4.0
error             // Error message string or null
isUploading       // Image upload in progress
isFetchingNutrition // Nutrition API call in progress
isSaving          // Meal save in progress
showFullNutrition // Full nutrition modal visible
```

---

## Code Structure

### Files Modified

#### `mobile/src/screens/ScanScreen.js` (Complete Rewrite)
- 670+ lines of production-grade React Native code
- Three-step flow withstate management
- Full error handling and loading states
- Comprehensive styling with safe areas
- Modal for full nutrition display

#### `mobile/src/api/client.js` (Enhanced)
- Added timeout-based fetch wrapper (8s regular, 15s upload)
- Better error messages with response text parsing
- Proper AbortController cleanup
- MIME type detection for image uploads
- Detailed console logging for debugging

### Files Unchanged
- `src/api/nutrition.py`: `/map/nutrition` API working as-is
- `src/api/food_mapper.py`: Edamam/USDA integration working as-is
- All backend endpoints functional

---

## Testing Checklist

### Setup
- [ ] Backend running: `python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000 --reload`
- [ ] Mobile app running: `expo start -c` (clear cache)
- [ ] Logged in with valid user account

### Scan Landing Page
- [ ] Page loads with proper safe area (content below status bar)
- [ ] Camera/Upload buttons side-by-side and tappable
- [ ] Tips card displays all 4 tips correctly
- [ ] Tab bar visible at bottom
- [ ] Dark mode colors applied correctly

### Camera/Upload Flow
- [ ] "Take Photo" button opens camera
- [ ] "Upload" button opens gallery
- [ ] Image preview appears in portions step
- [ ] Confidence badge displays on top-right (e.g., "97% match")

### Predictions
- [ ] All 3 predictions rendered with emoji, name, confidence
- [ ] Tapping a prediction selects it (border highlights)
- [ ] Selecting prediction fetches new nutrition data
- [ ] Loading spinner shows during nutrition fetch

### Portion Adjustment
- [ ] 4 preset buttons (Small/Medium/Large/Extra Large)
- [ ] Each button shows weight in grams
- [ ] Custom multiplier row shows: [- button] [1.00x display] [+ button]
- [ ] Pressing +/- changes multiplier by 0.25
- [ ] Multiplier clamped at 0.25 and 4.0 (can't go lower/higher)
- [ ] Macro values update in real-time as multiplier changes

### Nutrition Display
- [ ] 4-column macro grid shows: Calories, Protein, Carbs, Fat
- [ ] Values update dynamically when portion changes
- [ ] "View Full Nutrition Facts" button visible

### Full Nutrition Modal
- [ ] Modal appears with slide-up animation
- [ ] Header shows "Nutrition Facts" title with X button
- [ ] All 8 nutrients displayed (Calories, Protein, Carbs, Fiber, Sugar, Fat, Sodium, Cholesterol)
- [ ] Values match scaled nutrition
- [ ] Disclaimer text visible at bottom
- [ ] Close button works and dismisses modal
- [ ] X button in header dismisses modal

### Review Step
- [ ] Image preview shows (smaller than portions step)
- [ ] Meal name and portion size displayed
- [ ] 4-column macro grid shows scaled values
- [ ] "View Full Nutrition Label" button visible
- [ ] Back button returns to portions step
- [ ] "Add to Log" button appears

### Meal Save
- [ ] Clicking "Add to Log" shows loading spinner
- [ ] Backend receives `/map/log` POST with correct payload
- [ ] Success alert appears: "Meal saved to your log!"
- [ ] "View Log" button navigates to Home
- [ ] Home screen shows new meal in today's meals list
- [ ] Macro totals on Home update to include new meal

### Error Handling
- [ ] Upload with network disabled → Timeout error message
- [ ] Prediction with no matches → "Could not identify food..." error
- [ ] Nutrition lookup fails → "No nutrition data found..." error
- [ ] Save with no token → Auth error (handled by client)
- [ ] All errors have retry paths available

### Dark/Light Mode
- [ ] Colors apply correctly in light mode
- [ ] Colors apply correctly in dark mode
- [ ] Modal overlay visible in both modes
- [ ] Text contrast adequate in both modes

---

## Integration with Home Screen

After saving a meal, the Home screen should:
1. Fetch latest meals from `/api/mobile/dashboard`
2. Display new meal in "Today's Meals" section
3. Update calorie and macro totals
4. Update visual indicators (progress rings, etc.)

If Home doesn't auto-refresh:
1. Implement `useFocusEffect` hook to refresh on navigation focus
2. Or add pull-to-refresh gesture
3. Or use Redux/Context for shared meal state

---

## Future Enhancements

1. **Barcode Scanning**: Integrate barcode scanner for packaged foods
2. **Recipe Breakdown**: Auto-parse multi-ingredient meals
3. **Nutrition History**: View past scans and trends
4. **Favorites**: Quick-select frequently scanned meals
5. **Custom Meals**: Let users create meals not in API
6. **Receipt Parsing**: Extract nutrition from photos of labels
7. **Voice Input**: Dictate meal name instead of typing
8. **Offline Mode**: Cache recent predictions and nutrition data

---

## Performance Notes

- Image compression: 80% quality (~50-100KB typical)
- API timeout: 8-15 seconds (avoids indefinite hanging)
- Scroll performance: ScrollView contentContainerStyle with paddingBottom
- Modal animation: `animationType="slide"` for smooth appearance
- No unnecessary re-renders: useEffect dependencies properly specified

---

## Support & Debugging

**Logs to check:**
```javascript
console.error(`[GET ${endpoint}] Error:`, error.message);
console.error(`[POST ${endpoint}] Error:`, error.message);
console.error(`[UPLOAD ${endpoint}] Error:`, error.message);
console.warn('API /predict failed', err);
console.warn('Failed to fetch nutrition:', error);
console.error('Save failed:', err);
```

**Network Inspector:**
- Check `/predict` request payload (FormData with image file)
- Check `/map/nutrition` request/response
- Check `/map/log` request payload (JSON with nutrition data)

**Backend Health:**
- `GET /health` should return 200
- `/predict` endpoint available and model loaded
- Edamam/USDA API keys configured
- Meal logs being saved to `outputs/meal_logs.json`
