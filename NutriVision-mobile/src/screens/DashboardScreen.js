import React, { useEffect, useMemo, useState, useCallback } from 'react';
import {
  Animated,
  Alert,
  Image,
  KeyboardAvoidingView,
  LayoutAnimation,
  Modal,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  UIManager,
  View,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import * as ImagePicker from 'expo-image-picker';
import { useFocusEffect } from '@react-navigation/native';
import { useTheme } from '../theme/ThemeContext';
import { Typography, Spacing, BorderRadius, Shadow } from '../theme';
import { Button, Card, ProgressBar, ProgressRing } from '../components';
import { useMeals, toDateKey, DRINK_TYPES, getHydrationTarget } from '../context/MealContext';
import { useAuth } from '../context/AuthContext';
import { API_BASE_URL, apiRequest } from '../api/client';

const DEFAULT_PROFILE = {
  age: 23,
  gender: 'male',
  heightCm: 180,
  weightKg: 82,
  activityMultiplier: 1.2,
  goalType: 'maintain',
  interpretedGoal: 'Maintain Health',
  confidence: 'Low',
  planLabel: 'Low Carb, Halal',
};

const GOAL_CALORIE_ADJUSTMENT = {
  lose: -400,
  maintain: 0,
  gain: 300,
  muscle: 200,
};

const GOAL_MACRO_SPLIT = {
  lose: { protein: 0.35, carbs: 0.3, fat: 0.35 },
  maintain: { protein: 0.28, carbs: 0.42, fat: 0.3 },
  gain: { protein: 0.27, carbs: 0.48, fat: 0.25 },
  muscle: { protein: 0.33, carbs: 0.42, fat: 0.25 },
};

const GOAL_LABELS = {
  lose: 'Lose Weight',
  maintain: 'Maintain Weight',
  gain: 'Gain Weight',
  muscle: 'Gain Muscle',
};

const SAMPLE_MEALS = [
  {
    name: 'Chicken Rice Bowl',
    calories: 560,
    protein: 42,
    carbs: 52,
    fat: 18,
    nutrients: { VitaminA: 120, VitaminC: 24, VitaminD: 3, Calcium: 130, Magnesium: 80, Iron: 4, Choline: 90 },
  },
  {
    name: 'Salmon and Potatoes',
    calories: 640,
    protein: 45,
    carbs: 50,
    fat: 25,
    nutrients: { VitaminD: 10, VitaminE: 3, Calcium: 90, Potassium: 700, Omega3ALA: 0.2, Omega6LA: 0.6 },
  },
  {
    name: 'Greek Yogurt Fruit Bowl',
    calories: 390,
    protein: 24,
    carbs: 42,
    fat: 12,
    nutrients: { VitaminC: 18, Calcium: 290, Magnesium: 45, Zinc: 2.5, Choline: 70 },
  },
];

const NUTRIENT_TEMPLATE = {
  core: [
    { key: 'fiber', label: 'Fiber', target: 30, unit: 'g' },
    { key: 'choline', label: 'Choline', target: 450, unit: 'mg' },
    { key: 'water', label: 'Water', target: 2500, unit: 'ml' },
  ],
  vitamins: [
    { key: 'vitaminA', label: 'Vitamin A', target: 900, unit: 'mcg' },
    { key: 'vitaminC', label: 'Vitamin C', target: 90, unit: 'mg' },
    { key: 'vitaminD', label: 'Vitamin D', target: 15, unit: 'mcg' },
    { key: 'vitaminE', label: 'Vitamin E', target: 15, unit: 'mg' },
    { key: 'vitaminK', label: 'Vitamin K', target: 120, unit: 'mcg' },
    { key: 'vitaminB1', label: 'Vitamin B1 (Thiamine)', target: 1.2, unit: 'mg' },
    { key: 'vitaminB2', label: 'Vitamin B2 (Riboflavin)', target: 1.3, unit: 'mg' },
    { key: 'vitaminB3', label: 'Vitamin B3 (Niacin)', target: 16, unit: 'mg' },
    { key: 'vitaminB5', label: 'Vitamin B5 (Pantothenic acid)', target: 5, unit: 'mg' },
    { key: 'vitaminB6', label: 'Vitamin B6', target: 1.3, unit: 'mg' },
    { key: 'vitaminB7', label: 'Vitamin B7 (Biotin)', target: 30, unit: 'mcg' },
    { key: 'vitaminB9', label: 'Vitamin B9 (Folate)', target: 400, unit: 'mcg' },
    { key: 'vitaminB12', label: 'Vitamin B12', target: 2.4, unit: 'mcg' },
  ],
  majorMinerals: [
    { key: 'calcium', label: 'Calcium', target: 1000, unit: 'mg' },
    { key: 'phosphorus', label: 'Phosphorus', target: 700, unit: 'mg' },
    { key: 'magnesium', label: 'Magnesium', target: 420, unit: 'mg' },
    { key: 'sodium', label: 'Sodium', target: 2300, unit: 'mg' },
    { key: 'potassium', label: 'Potassium', target: 3400, unit: 'mg' },
    { key: 'chloride', label: 'Chloride', target: 2300, unit: 'mg' },
    { key: 'sulfur', label: 'Sulfur', target: 900, unit: 'mg' },
  ],
  traceMinerals: [
    { key: 'iron', label: 'Iron', target: 18, unit: 'mg' },
    { key: 'zinc', label: 'Zinc', target: 11, unit: 'mg' },
    { key: 'iodine', label: 'Iodine', target: 150, unit: 'mcg' },
    { key: 'selenium', label: 'Selenium', target: 55, unit: 'mcg' },
    { key: 'copper', label: 'Copper', target: 0.9, unit: 'mg' },
    { key: 'manganese', label: 'Manganese', target: 2.3, unit: 'mg' },
    { key: 'fluoride', label: 'Fluoride', target: 4, unit: 'mg' },
    { key: 'chromium', label: 'Chromium', target: 35, unit: 'mcg' },
    { key: 'molybdenum', label: 'Molybdenum', target: 45, unit: 'mcg' },
  ],
  aminoAcids: [
    { key: 'histidine', label: 'Histidine', target: 900, unit: 'mg' },
    { key: 'isoleucine', label: 'Isoleucine', target: 1300, unit: 'mg' },
    { key: 'leucine', label: 'Leucine', target: 3000, unit: 'mg' },
    { key: 'lysine', label: 'Lysine', target: 2500, unit: 'mg' },
    { key: 'methionineCysteine', label: 'Methionine + Cysteine', target: 1300, unit: 'mg' },
    { key: 'phenylalanineTyrosine', label: 'Phenylalanine + Tyrosine', target: 2200, unit: 'mg' },
    { key: 'threonine', label: 'Threonine', target: 1400, unit: 'mg' },
    { key: 'tryptophan', label: 'Tryptophan', target: 350, unit: 'mg' },
    { key: 'valine', label: 'Valine', target: 1600, unit: 'mg' },
  ],
  fattyAcids: [
    { key: 'omega3', label: 'Omega-3 (ALA)', target: 1.6, unit: 'g' },
    { key: 'omega6', label: 'Omega-6 (Linoleic acid)', target: 17, unit: 'g' },
  ],
};

const CATEGORY_TITLES = {
  core: 'Other Important Nutrients',
  vitamins: 'Vitamins',
  majorMinerals: 'Major Minerals',
  traceMinerals: 'Trace Minerals',
  aminoAcids: 'Essential Amino Acids',
  fattyAcids: 'Fatty Acids',
};

const NUTRIENT_INFO = {
  vitaminD: {
    purpose: 'Supports bone health and immune function.',
    sources: 'Salmon, fortified dairy, egg yolks.',
  },
  magnesium: {
    purpose: 'Supports muscle and nerve function.',
    sources: 'Nuts, seeds, legumes, leafy greens.',
  },
  vitaminA: {
    purpose: 'Supports vision and immune defense.',
    sources: 'Carrot, sweet potato, spinach, eggs.',
  },
};

function safeParseFloat(value, fallback) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function getGreeting() {
  const hour = new Date().getHours();
  if (hour < 12) return 'Good Morning';
  if (hour < 17) return 'Good Afternoon';
  return 'Good Evening';
}

function toMealTimeLabel(date) {
  return date.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
}

function inferGoalType(goalText = '') {
  const lower = goalText.toLowerCase();
  if (lower.includes('muscle') || lower.includes('lean') || lower.includes('bulk')) return 'muscle';
  if (lower.includes('lose') || lower.includes('fat')) return 'lose';
  if (lower.includes('gain') || lower.includes('build')) return 'gain';
  return 'maintain';
}

function toGoalLabel(goalType) {
  return GOAL_LABELS[goalType] || GOAL_LABELS.maintain;
}

function computeMacroTargets(goalType, calories, weightKg) {
  const split = GOAL_MACRO_SPLIT[goalType] || GOAL_MACRO_SPLIT.maintain;
  return {
    protein: Math.round((calories * split.protein) / 4),
    carbs: Math.round((calories * split.carbs) / 4),
    fat: Math.round((calories * split.fat) / 9),
    water: Number((weightKg * 0.04).toFixed(1)),
  };
}

function calculateBmr({ age, gender, heightCm, weightKg }) {
  const base = 10 * weightKg + 6.25 * heightCm - 5 * age;
  return gender.toLowerCase() === 'male' ? base + 5 : base - 161;
}

function createDefaultNutrientTargets(targetSource = null) {
  const targets = {};
  Object.values(NUTRIENT_TEMPLATE).flat().forEach((item) => {
    const dynamicTarget = Number(targetSource?.[item.key]);
    targets[item.key] = Number.isFinite(dynamicTarget) && dynamicTarget > 0 ? dynamicTarget : item.target;
  });
  return targets;
}

function createDefaultNutrientTargetModes() {
  const modes = {};
  Object.values(NUTRIENT_TEMPLATE).flat().forEach((item) => {
    modes[item.key] = 'auto';
  });
  return modes;
}

function getNutrientProgress(consumedValue, targetValue) {
  if (!Number.isFinite(targetValue) || targetValue <= 0) return 0;
  return Math.max(0, Math.min(consumedValue / targetValue, 1.5));
}

function getNutrientStatus(consumedValue, targetValue) {
  if (!Number.isFinite(consumedValue) || consumedValue <= 0) return 'low';
  if (!Number.isFinite(targetValue) || targetValue <= 0) return 'on track';
  const ratio = consumedValue / targetValue;
  if (ratio > 1.05) return 'over target';
  if (ratio >= 0.75) return 'on track';
  return 'low';
}

export default function DashboardScreen({ navigation, route }) {
  const { colors } = useTheme();
  const insets = useSafeAreaInsets();
  const {
    addMeal: addMealToContext,
    removeMeal: removeMealFromContext,
    getMealsForDate,
    getTotalsForDate,
    getDailyXpForDate,
    getXpProgression,
    loadMeals,
    personalizedTargets,
    saveMealTemplate,
    toggleFavoriteMeal,
    getFavoriteMeals,
    getRecentMeals,
    deleteSavedMeal,
    logSavedMeal,
    savedMeals,
    favoriteMealIds,
    logHydration: logHydrationFromContext,
    getHydrationForDate,
    getHydrationNutrientsForDate,
  } = useMeals();
  const { profile, updateProfile, token } = useAuth();
  const onboardingData = route?.params?.onboardingData || {};

  const startingGoalText = onboardingData.customGoal || onboardingData.selectedGoals?.[0] || '';

  const [profileSettings, setProfileSettings] = useState(() => {
    const personalInfo = onboardingData.personalInfo || {};
    const age = safeParseFloat(personalInfo.age, DEFAULT_PROFILE.age);
    const weightKg = safeParseFloat(personalInfo.weightLb, DEFAULT_PROFILE.weightKg * 2.2046226218) / 2.2046226218;
    const heightCm = safeParseFloat(personalInfo.heightCm, DEFAULT_PROFILE.heightCm);
    const activityMultiplier = safeParseFloat(onboardingData.activityMultiplier, DEFAULT_PROFILE.activityMultiplier);
    const goalType = inferGoalType(startingGoalText || DEFAULT_PROFILE.goalType);

    return {
      age,
      gender: (personalInfo.gender || DEFAULT_PROFILE.gender).toLowerCase(),
      heightCm,
      weightKg,
      activityMultiplier,
      goalType,
      interpretedGoal: DEFAULT_PROFILE.interpretedGoal,
      confidence: DEFAULT_PROFILE.confidence,
      planLabel: joinPlan(onboardingData.selectedDietaryPreferences),
    };
  });

  const [customCalories, setCustomCalories] = useState(null);
  const todayKey = toDateKey(new Date());
  const meals = getMealsForDate(todayKey);
  const [macroTargets, setMacroTargets] = useState(() => ({
    protein: Number(personalizedTargets?.targets?.protein) || 120,
    carbs: Number(personalizedTargets?.targets?.carbs) || 250,
    fat: Number(personalizedTargets?.targets?.fat) || 65,
    water: Number((Number(personalizedTargets?.targets?.water || 3200) / 1000).toFixed(1)),
  }));
  const [nutrientTargets, setNutrientTargets] = useState(() => createDefaultNutrientTargets(personalizedTargets?.targets));
  const [nutrientTargetModes, setNutrientTargetModes] = useState(createDefaultNutrientTargetModes);
  const [isCalorieModalOpen, setIsCalorieModalOpen] = useState(false);
  const [isMacroModalOpen, setIsMacroModalOpen] = useState(false);
  const [isScanModalOpen, setIsScanModalOpen] = useState(false);
  const [scanModalTab, setScanModalTab] = useState('scan');
  const [selectedMeal, setSelectedMeal] = useState(null);
  const [isInsightsExpanded, setIsInsightsExpanded] = useState(false);
  const [viewAllCategory, setViewAllCategory] = useState(null);
  const [editingNutrient, setEditingNutrient] = useState(null);
  const [editingCategoryTargets, setEditingCategoryTargets] = useState(null);
  const [infoNutrient, setInfoNutrient] = useState(null);
  const [nutrientEditValue, setNutrientEditValue] = useState('');
  const [goalDraft, setGoalDraft] = useState(profileSettings.goalType);
  const [activityDraft, setActivityDraft] = useState(profileSettings.activityMultiplier);
  const [savingGoal, setSavingGoal] = useState(false);
  const [goalError, setGoalError] = useState('');
  const [targetError, setTargetError] = useState('');
  const [isWaterModalOpen, setIsWaterModalOpen] = useState(false);
  const [customWaterAmount, setCustomWaterAmount] = useState('');
  const [customWaterUnit, setCustomWaterUnit] = useState('ml');
  const [selectedDrinkType, setSelectedDrinkType] = useState('plain');
  const [isMealEntryModalOpen, setIsMealEntryModalOpen] = useState(false);
  const [isSavedMealsBrowserOpen, setIsSavedMealsBrowserOpen] = useState(false);
  const [savedMealBrowserMode, setSavedMealBrowserMode] = useState('saved'); // 'saved', 'favorites', 'recent'
  const [selectedSavedMealForLog, setSelectedSavedMealForLog] = useState(null);
  const [isPortionAdjustOpen, setIsPortionAdjustOpen] = useState(false);
  const [portionMultiplier, setPortionMultiplier] = useState(1);
  const [portionInputUnit, setPortionInputUnit] = useState('x');
  const [portionInputValue, setPortionInputValue] = useState('1');
  const [selectedMealForAction, setSelectedMealForAction] = useState(null);
  const [isMealActionModalOpen, setIsMealActionModalOpen] = useState(false);
  const [isDailyXpExpanded, setIsDailyXpExpanded] = useState(false);

  useEffect(() => {
    if (Platform.OS === 'android' && UIManager.setLayoutAnimationEnabledExperimental) {
      UIManager.setLayoutAnimationEnabledExperimental(true);
    }
  }, []);

  useEffect(() => {
    if (!personalizedTargets?.targets) return;
    setNutrientTargets((prev) => ({
      ...createDefaultNutrientTargets(personalizedTargets.targets),
      ...prev,
    }));
    setMacroTargets((prev) => ({
      ...prev,
      protein: Number(personalizedTargets.targets.protein || prev.protein),
      carbs: Number(personalizedTargets.targets.carbs || prev.carbs),
      fat: Number(personalizedTargets.targets.fat || prev.fat),
      water: Number((Number(personalizedTargets.targets.water || prev.water * 1000) / 1000).toFixed(1)),
    }));
  }, [personalizedTargets]);

  // Reload meals from backend every time this screen comes into focus
  // Limit to 30 meals for faster loading (covers ~30-40 days of history)
  useFocusEffect(
    useCallback(() => {
      if (!token) return;
      apiRequest('/map/logs?limit=30', { token })
        .then((data) => {
          if (data?.entries) loadMeals(data.entries);
        })
        .catch(() => {});
    }, [token, loadMeals]),
  );

  useEffect(() => {
    if (!profile) return;

    const incomingGoalType = ['lose', 'maintain', 'gain', 'muscle'].includes(profile?.goalType)
      ? profile.goalType
      : inferGoalType(profile?.goal || '');
    const incomingActivity = Number(profile?.activityMultiplier);
    const incomingCalories = Number(profile?.dailyCalorieGoal);
    const incomingProtein = Number(profile?.proteinGoal);
    const incomingCarbs = Number(profile?.carbsGoal);
    const incomingFat = Number(profile?.fatGoal);
    const incomingNutrientTargets = profile?.nutrientTargets && typeof profile.nutrientTargets === 'object'
      ? profile.nutrientTargets
      : null;
    const incomingNutrientModes = profile?.nutrientTargetModes && typeof profile.nutrientTargetModes === 'object'
      ? profile.nutrientTargetModes
      : null;

    setProfileSettings((prev) => ({
      ...prev,
      goalType: incomingGoalType || prev.goalType,
      interpretedGoal: toGoalLabel(incomingGoalType || prev.goalType),
      activityMultiplier: Number.isFinite(incomingActivity) ? incomingActivity : prev.activityMultiplier,
    }));

    if (!isCalorieModalOpen) {
      setGoalDraft(incomingGoalType || 'maintain');
      if (Number.isFinite(incomingActivity)) {
        setActivityDraft(incomingActivity);
      }
      if (Number.isFinite(incomingCalories) && incomingCalories >= 1200) {
        setCustomCalories(incomingCalories);
      }
    }
    if (Number.isFinite(incomingProtein) && Number.isFinite(incomingCarbs) && Number.isFinite(incomingFat)) {
      setMacroTargets((prev) => ({
        ...prev,
        protein: Math.max(20, Math.round(incomingProtein)),
        carbs: Math.max(20, Math.round(incomingCarbs)),
        fat: Math.max(20, Math.round(incomingFat)),
      }));
    }

    if (incomingNutrientTargets) {
      setNutrientTargets((prev) => ({
        ...prev,
        ...Object.fromEntries(
          Object.entries(incomingNutrientTargets).map(([key, value]) => [key, Number(value)])
        ),
      }));
    }

    if (incomingNutrientModes) {
      setNutrientTargetModes((prev) => ({
        ...prev,
        ...Object.fromEntries(
          Object.entries(incomingNutrientModes).map(([key, value]) => [key, String(value || '').toLowerCase() === 'custom' ? 'custom' : 'auto'])
        ),
      }));
    }
  }, [isCalorieModalOpen, profile]);

  const bmr = useMemo(
    () => Number(personalizedTargets?.bmr) || calculateBmr(profileSettings),
    [personalizedTargets, profileSettings],
  );
  const tdee = useMemo(
    () => Number(personalizedTargets?.tdee) || bmr * profileSettings.activityMultiplier,
    [bmr, personalizedTargets, profileSettings.activityMultiplier],
  );
  const recommendedCalories = useMemo(
    () => Math.max(1200, Math.round(tdee + GOAL_CALORIE_ADJUSTMENT[profileSettings.goalType])),
    [profileSettings.goalType, tdee],
  );

  const recommendedDraftCalories = useMemo(() => {
    const draftTdee = bmr * activityDraft;
    return Math.max(1200, Math.round(draftTdee + GOAL_CALORIE_ADJUSTMENT[goalDraft]));
  }, [activityDraft, bmr, goalDraft]);

  const calorieGoal = Number(profile?.dailyCalorieGoal) || customCalories || Number(personalizedTargets?.targets?.calories) || recommendedCalories;

  const dailyTotals = useMemo(() => getTotalsForDate(todayKey), [getTotalsForDate, todayKey]);

  const consumed = useMemo(
    () => ({
      calories: dailyTotals.calories,
      protein: dailyTotals.protein,
      carbs: dailyTotals.carbs,
      fat: dailyTotals.fat,
    }),
    [dailyTotals],
  );

  const nutrientConsumed = useMemo(
    () =>
      Object.values(NUTRIENT_TEMPLATE)
        .flat()
        .reduce((acc, nutrient) => {
          acc[nutrient.key] = Number(dailyTotals[nutrient.key] || 0);
          return acc;
        }, {}),
    [dailyTotals],
  );

  const caloriesRemaining = Math.max(0, calorieGoal - consumed.calories);
  const caloriesProgress = calorieGoal > 0 ? Math.min(consumed.calories / calorieGoal, 1) : 0;
  const usedPercent = calorieGoal > 0 ? Math.round((consumed.calories / calorieGoal) * 100) : 0;

  const autoMacroTargets = useMemo(
    () => computeMacroTargets(profileSettings.goalType, calorieGoal, profileSettings.weightKg),
    [calorieGoal, profileSettings.goalType, profileSettings.weightKg],
  );

  const macroData = [
    {
      key: 'protein',
      label: 'PROTEIN',
      consumed: consumed.protein,
      target: macroTargets.protein,
      color: '#8b5cf6',
    },
    {
      key: 'carbs',
      label: 'CARBS',
      consumed: consumed.carbs,
      target: macroTargets.carbs,
      color: '#facc15',
    },
    {
      key: 'fat',
      label: 'FAT',
      consumed: consumed.fat,
      target: macroTargets.fat,
      color: '#ef4444',
    },
  ];

  const insightsSummary = useMemo(
    () => Object.entries(CATEGORY_TITLES).map(([key, title]) => ({
      key,
      title,
      description: `${(NUTRIENT_TEMPLATE[key] || []).length} nutrients`,
    })),
    [],
  );

  const nutrientSummaryCounts = useMemo(() => {
    const allNutrients = Object.values(NUTRIENT_TEMPLATE).flat();
    return allNutrients.reduce(
      (acc, nutrient) => {
        const target = Number(nutrientTargets[nutrient.key] ?? nutrient.target);
        const consumedValue = Number(nutrientConsumed[nutrient.key] || 0);
        const status = getNutrientStatus(consumedValue, target);
        if (status === 'low') acc.low += 1;
        if (status === 'on track') acc.onTrack += 1;
        if (status === 'over target') acc.over += 1;
        if ((nutrientTargetModes[nutrient.key] || 'auto') === 'custom') acc.custom += 1;
        return acc;
      },
      { low: 0, onTrack: 0, over: 0, custom: 0 },
    );
  }, [nutrientConsumed, nutrientTargetModes, nutrientTargets]);

  const hydrationConsumedMl = useMemo(() => Math.round(getHydrationForDate(todayKey) || 0), [getHydrationForDate, todayKey]);
  const hydrationTargetMl = useMemo(() => getHydrationTarget(profileSettings), [profileSettings]);
  const hydrationProgress = useMemo(() => {
    if (!Number.isFinite(hydrationTargetMl) || hydrationTargetMl <= 0) return 0;
    return Math.min(hydrationConsumedMl / hydrationTargetMl, 1);
  }, [hydrationConsumedMl, hydrationTargetMl]);
  const hydrationProgressAnim = React.useRef(new Animated.Value(hydrationProgress)).current;
  const hydrationProgressWidth = hydrationProgressAnim.interpolate({
    inputRange: [0, 1],
    outputRange: ['0%', '100%'],
  });

  useEffect(() => {
    Animated.timing(hydrationProgressAnim, {
      toValue: hydrationProgress,
      duration: 320,
      useNativeDriver: false,
    }).start();
  }, [hydrationProgress, hydrationProgressAnim]);

  const hydrationNutrients = useMemo(() => getHydrationNutrientsForDate(todayKey) || {}, [getHydrationNutrientsForDate, todayKey]);
  const hydrationMineralItems = useMemo(
    () => [
      { key: 'sodium', label: 'Na', value: Number(hydrationNutrients.sodium || 0) },
      { key: 'potassium', label: 'K', value: Number(hydrationNutrients.potassium || 0) },
      { key: 'magnesium', label: 'Mg', value: Number(hydrationNutrients.magnesium || 0) },
      { key: 'calcium', label: 'Ca', value: Number(hydrationNutrients.calcium || 0) },
      { key: 'chloride', label: 'Cl', value: Number(hydrationNutrients.chloride || 0) },
    ].filter((item) => item.value > 0),
    [hydrationNutrients],
  );

  const favoriteMeals = useMemo(() => getFavoriteMeals(), [getFavoriteMeals]);
  const dailyXp = useMemo(() => getDailyXpForDate(todayKey), [getDailyXpForDate, todayKey]);
  const xpProgression = useMemo(() => getXpProgression(), [getXpProgression]);
  const xpCategoryRows = useMemo(
    () => [
      { key: 'calories', label: 'Calories' },
      { key: 'macros', label: 'Macros' },
      { key: 'fiber', label: 'Fiber' },
      { key: 'choline', label: 'Choline' },
      { key: 'water', label: 'Water' },
      { key: 'vitamins', label: 'Vitamins' },
      { key: 'majorMinerals', label: 'Major Minerals' },
      { key: 'traceMinerals', label: 'Trace Minerals' },
      { key: 'aminoAcids', label: 'Amino Acids' },
      { key: 'fattyAcids', label: 'Fatty Acids' },
      { key: 'bonus', label: 'Bonus' },
    ],
    [],
  );
  const xpReasonPreview = useMemo(() => (dailyXp?.reasons || []).slice(0, 6), [dailyXp]);
  const xpMissedRows = useMemo(() => {
    const breakdown = dailyXp?.breakdown || {};
    const maxByCategory = dailyXp?.maxByCategory || {};
    return xpCategoryRows
      .map((row) => {
        const earned = Number(breakdown[row.key] || 0);
        const max = Number(maxByCategory[row.key] || 0);
        const missed = Math.max(0, max - earned);
        return { ...row, earned, max, missed };
      })
      .filter((row) => row.max > 0 && row.missed > 0)
      .sort((a, b) => b.missed - a.missed)
      .slice(0, 5);
  }, [dailyXp, xpCategoryRows]);

  const toggleDailyXpExpanded = () => {
    LayoutAnimation.configureNext({
      duration: 240,
      create: { type: 'easeInEaseOut', property: 'opacity' },
      update: { type: 'easeInEaseOut' },
      delete: { type: 'easeInEaseOut', property: 'opacity' },
    });
    setIsDailyXpExpanded((prev) => !prev);
  };

  const findTemplateForMeal = useCallback(
    (meal) => {
      if (!meal) return null;
      const bySource = savedMeals.find((m) => m.sourceId && meal.id && m.sourceId === meal.id);
      if (bySource) return bySource;
      return savedMeals.find(
        (m) =>
          m.name === meal.name
          && Math.abs(Number(m.calories || 0) - Number(meal.calories || 0)) <= 5
          && Math.abs(Number(m.protein || 0) - Number(meal.protein || 0)) <= 2
          && Math.abs(Number(m.carbs || 0) - Number(meal.carbs || 0)) <= 2
          && Math.abs(Number(m.fat || 0) - Number(meal.fat || 0)) <= 2,
      ) || null;
    },
    [savedMeals],
  );

  const selectedMealTemplate = useMemo(() => findTemplateForMeal(selectedMeal), [findTemplateForMeal, selectedMeal]);
  const selectedMealIsFavorite = useMemo(
    () => Boolean(selectedMealTemplate && favoriteMealIds.has(selectedMealTemplate.templateId)),
    [favoriteMealIds, selectedMealTemplate],
  );

  const effectivePortionMultiplier = useMemo(() => {
    const xValue = Number(portionInputValue);
    if (portionInputUnit === 'x') {
      if (Number.isFinite(xValue) && xValue > 0) return xValue;
      return portionMultiplier;
    }

    const baseServing = String(selectedSavedMealForLog?.servingSize || '').toLowerCase();
    let baseGrams = 100;
    const numeric = Number(baseServing.replace(/[^0-9.]/g, ''));
    if (Number.isFinite(numeric) && numeric > 0) {
      if (baseServing.includes('oz')) baseGrams = numeric * 28.3495;
      else baseGrams = numeric;
    }

    if (!Number.isFinite(xValue) || xValue <= 0) return 1;
    const grams = portionInputUnit === 'g' ? xValue : xValue * 28.3495;
    return Math.max(0.1, Number((grams / baseGrams).toFixed(2)));
  }, [portionInputUnit, portionInputValue, portionMultiplier, selectedSavedMealForLog]);

  const addMeal = (mealBase, imageUri = null) => {
    addMealToContext(mealBase, imageUri);
    setIsScanModalOpen(false);
  };

  const handlePickMealImage = async () => {
    const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!permission.granted) return;

    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      quality: 0.85,
      allowsEditing: true,
    });

    if (!result.canceled && result.assets && result.assets[0]?.uri) {
      const randomMeal = SAMPLE_MEALS[Math.floor(Math.random() * SAMPLE_MEALS.length)];
      addMeal({ ...randomMeal, name: `${randomMeal.name} (AI)` }, result.assets[0].uri);
    }
  };

  const removeMeal = async (mealId) => {
    try {
      await removeMealFromContext(mealId, token);
      setSelectedMeal(null);
    } catch (err) {
      Alert.alert('Delete Failed', err?.message || 'Could not delete meal. Please try again.');
    }
  };

  const addWater = (amount, unit = 'ml', drinkType = 'plain') => {
    const waterAmount = Number(amount);
    if (!Number.isFinite(waterAmount) || waterAmount <= 0) {
      Alert.alert('Invalid Input', 'Please enter a valid water amount.');
      return;
    }

    try {
      logHydrationFromContext(waterAmount, unit, drinkType, {});

      setIsWaterModalOpen(false);
      setCustomWaterAmount('');
      setCustomWaterUnit('ml');
      setSelectedDrinkType('plain');
    } catch (err) {
      Alert.alert('Error', 'Could not log hydration. Please try again.');
    }
  };

  const handleSavedMealSelected = (savedMeal) => {
    setSelectedSavedMealForLog(savedMeal);
    setPortionMultiplier(1);
    setPortionInputUnit('x');
    setPortionInputValue('1');
    setIsSavedMealsBrowserOpen(false);
    setIsPortionAdjustOpen(true);
  };

  const handleLogSavedMealWithPortion = () => {
    if (!selectedSavedMealForLog) return;
    const loggedMeal = logSavedMeal(selectedSavedMealForLog, effectivePortionMultiplier);
    if (loggedMeal) {
      setIsPortionAdjustOpen(false);
      setSelectedSavedMealForLog(null);
      setPortionMultiplier(1);
      setPortionInputUnit('x');
      setPortionInputValue('1');
      Alert.alert('Success', `${selectedSavedMealForLog.name} logged!`);
    }
  };

  const handleToggleFavoriteFromSelectedMeal = () => {
    if (!selectedMeal) return;
    const existing = findTemplateForMeal(selectedMeal);

    if (existing) {
      const isFavorite = favoriteMealIds.has(existing.templateId);
      toggleFavoriteMeal(existing.templateId);
      Alert.alert(isFavorite ? 'Removed from Favorites' : 'Saved to Favorites', selectedMeal.name);
      return;
    }

    const template = saveMealTemplate(selectedMeal, selectedMeal.name);
    if (template) {
      toggleFavoriteMeal(template.templateId);
      Alert.alert('Saved to Favorites', selectedMeal.name);
    }
  };

  const handleSaveSampleMealToFavorites = (sampleMeal) => {
    const existing = findTemplateForMeal(sampleMeal);
    if (existing) {
      if (!favoriteMealIds.has(existing.templateId)) {
        toggleFavoriteMeal(existing.templateId);
      }
      Alert.alert('Saved to Favorites', sampleMeal.name);
      return;
    }

    const template = saveMealTemplate(sampleMeal, sampleMeal.name);
    if (template) {
      toggleFavoriteMeal(template.templateId);
      Alert.alert('Saved to Favorites', sampleMeal.name);
    }
  };

  const handleSaveMealAsTemplate = (mealId) => {
    if (!mealId || !meals.length) return;
    const meal = meals.find((m) => m.id === mealId);
    if (!meal) return;
    Alert.prompt('Save as Template', 'Name for this meal template:', [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Save',
        onPress: (name) => {
          if (name && name.trim()) {
            saveMealTemplate(meal, name.trim());
            Alert.alert('Saved!', `Meal saved as "${name.trim()}"`);
          }
        },
      },
    ]);
  };

  const handleToggleFavorite = (templateId) => {
    toggleFavoriteMeal(templateId);
  };

  const changeMacro = (field, delta) => {
    setMacroTargets((prev) => {
      const next = { ...prev };
      const min = field === 'water' ? 0.5 : 20;
      const step = field === 'water' ? 0.1 : 5;
      const rawValue = (prev[field] || 0) + delta * step;
      next[field] = field === 'water' ? Number(Math.max(min, rawValue).toFixed(1)) : Math.max(min, Math.round(rawValue));
      return next;
    });
  };

  const openNutrientEdit = (nutrient) => {
    setEditingNutrient(nutrient);
    setNutrientEditValue(String(nutrientTargets[nutrient.key] ?? nutrient.target));
    setTargetError('');
  };

  const saveNutrientEdit = async () => {
    if (!editingNutrient) return;
    const parsed = Number(nutrientEditValue);
    if (!Number.isFinite(parsed) || parsed <= 0) {
      setTargetError('Target must be a positive number.');
      return;
    }

    const nextTargets = { ...nutrientTargets, [editingNutrient.key]: parsed };
    const nextModes = { ...nutrientTargetModes, [editingNutrient.key]: 'custom' };

    setNutrientTargets(nextTargets);
    setNutrientTargetModes(nextModes);
    setEditingNutrient(null);
    setNutrientEditValue('');
    setTargetError('');

    try {
      await updateProfile({
        nutrientTargets: nextTargets,
        nutrientTargetModes: nextModes,
      });
    } catch (error) {
      setTargetError(error?.message || 'Unable to save nutrient target.');
    }
  };

  const resetNutrientTargetToAuto = async (nutrientKey) => {
    const defaults = createDefaultNutrientTargets(personalizedTargets?.targets);
    const nextTargets = { ...nutrientTargets, [nutrientKey]: defaults[nutrientKey] };
    const nextModes = { ...nutrientTargetModes, [nutrientKey]: 'auto' };
    setNutrientTargets(nextTargets);
    setNutrientTargetModes(nextModes);
    try {
      await updateProfile({ nutrientTargets: nextTargets, nutrientTargetModes: nextModes });
    } catch (error) {
      setTargetError(error?.message || 'Unable to reset nutrient target.');
    }
  };

  const resetCategoryTargetsToAuto = async (categoryKey) => {
    const defaults = createDefaultNutrientTargets(personalizedTargets?.targets);
    const nextTargets = { ...nutrientTargets };
    const nextModes = { ...nutrientTargetModes };

    (NUTRIENT_TEMPLATE[categoryKey] || []).forEach((item) => {
      nextTargets[item.key] = defaults[item.key];
      nextModes[item.key] = 'auto';
    });

    setNutrientTargets(nextTargets);
    setNutrientTargetModes(nextModes);

    try {
      await updateProfile({ nutrientTargets: nextTargets, nutrientTargetModes: nextModes });
    } catch (error) {
      setTargetError(error?.message || 'Unable to reset nutrient targets.');
    }
  };

  const applyGoalAndActivity = async () => {
    const goalType = ['lose', 'maintain', 'gain', 'muscle'].includes(goalDraft) ? goalDraft : 'maintain';
    const nextActivity = Number.isFinite(activityDraft) ? Math.min(2.4, Math.max(1.1, Number(activityDraft.toFixed(2)))) : 1.2;
    const targetCalories = Math.max(1200, Math.round(customCalories ?? recommendedDraftCalories));
    const nextMacros = computeMacroTargets(goalType, targetCalories, profileSettings.weightKg);

    setGoalError('');
    setSavingGoal(true);
    try {
      await updateProfile({
        goalType,
        goal: toGoalLabel(goalType),
        activityMultiplier: nextActivity,
        dailyCalorieGoal: targetCalories,
        proteinGoal: nextMacros.protein,
        carbsGoal: nextMacros.carbs,
        fatGoal: nextMacros.fat,
      });

      setProfileSettings((prev) => ({
        ...prev,
        goalType,
        interpretedGoal: toGoalLabel(goalType),
        activityMultiplier: nextActivity,
      }));
      setCustomCalories(targetCalories);
      setMacroTargets(nextMacros);
      setIsCalorieModalOpen(false);
    } catch (error) {
      setGoalError(error?.message || 'Unable to save goal right now.');
    } finally {
      setSavingGoal(false);
    }
  };

  const goalFocusLabel = profile?.goal || toGoalLabel(profileSettings.goalType);

  const renderNutrientCards = (categoryKey, limit = null) => {
    const list = NUTRIENT_TEMPLATE[categoryKey] || [];
    const visible = limit ? list.slice(0, limit) : list;

    return visible.map((item) => {
      const target = nutrientTargets[item.key] ?? item.target;
      const consumedValue = nutrientConsumed[item.key] || 0;
      const remaining = Math.max(0, target - consumedValue);
      const progress = getNutrientProgress(consumedValue, target);
      const status = getNutrientStatus(consumedValue, target);
      const mode = nutrientTargetModes[item.key] || 'auto';
      const percent = Math.max(0, Math.round((consumedValue / (target || 1)) * 100));

      const statusColor = status === 'over target'
        ? colors.danger
        : status === 'on track'
          ? colors.primary
          : colors.textSecondary;

      return (
        <View key={item.key} style={[styles.nutrientCard, { backgroundColor: colors.surfaceSecondary }]}> 
          <View style={styles.nutrientHeader}>
            <Text style={[styles.nutrientTitle, { color: colors.primary }]}>{item.label}</Text>
            <View style={styles.nutrientMetaRight}>
              <View style={[styles.modePill, { backgroundColor: mode === 'custom' ? colors.primarySoft : colors.surface }]}>
                <Text style={[styles.nutrientAuto, { color: mode === 'custom' ? colors.primary : colors.textSecondary }]}>{mode.toUpperCase()}</Text>
              </View>
              <TouchableOpacity onPress={() => setInfoNutrient(item)} activeOpacity={0.8} style={[styles.infoIconBtn, { borderColor: colors.border }]}> 
                <Text style={[styles.infoIconText, { color: colors.textSecondary }]}>i</Text>
              </TouchableOpacity>
            </View>
          </View>

          <View style={styles.nutrientStatusRow}>
            <Text style={[styles.nutrientStatusText, { color: statusColor }]}>{status.toUpperCase()}</Text>
            <Text style={[styles.nutrientPercentText, { color: colors.textSecondary }]}>{percent}% of target reached</Text>
          </View>

          <Text style={[styles.nutrientNumbers, { color: colors.textSecondary }]}>
            CONSUMED {formatAmount(consumedValue)} {item.unit.toUpperCase()}  |  REMAINING {formatAmount(remaining)} {item.unit.toUpperCase()}
          </Text>

          <ProgressBar progress={progress} color={statusColor} height={6} style={styles.nutrientProgressBar} />

          <View style={styles.nutrientBottomRow}>
            <Text style={[styles.nutrientTarget, { color: colors.textTertiary }]}>TARGET {formatAmount(target)} {item.unit.toUpperCase()}</Text>
            <View style={styles.nutrientActionRow}>
              {mode === 'custom' ? (
                <TouchableOpacity onPress={() => resetNutrientTargetToAuto(item.key)} activeOpacity={0.8}>
                  <Text style={[styles.nutrientReset, { color: colors.textSecondary }]}>RESET</Text>
                </TouchableOpacity>
              ) : null}
              <TouchableOpacity onPress={() => openNutrientEdit(item)} activeOpacity={0.8}>
                <Text style={[styles.nutrientEdit, { color: colors.primary }]}>EDIT</Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      );
    });
  };

  const profileName = profile?.name || 'User';
  const profileInitials = profile?.initials || 'U';
  const profileAvatar = profile?.avatar
    ? { uri: profile.avatar.startsWith('/') ? `${API_BASE_URL}${profile.avatar}` : profile.avatar }
    : null;

  return (
    <KeyboardAvoidingView style={[styles.container, { backgroundColor: colors.background }]} behavior={Platform.OS === 'ios' ? 'padding' : undefined}>
      <ScrollView
        style={[styles.container, { backgroundColor: colors.background }]}
        contentContainerStyle={[styles.content, { paddingTop: insets.top + 8, paddingBottom: insets.bottom + 72 }]}
        showsVerticalScrollIndicator={false}
      >
        <View style={styles.header}>
          <View>
            <Text style={[styles.greeting, { color: colors.textSecondary }]}>{getGreeting()}</Text>
            <Text style={[styles.userName, { color: colors.text }]}>{profileName.split(' ')[0]}</Text>
            <Text style={[styles.greetingSub, { color: colors.textSecondary }]}>Let's make today count.</Text>
          </View>
          <View style={styles.headerActions}>
            <TouchableOpacity style={[styles.iconBtn, { borderColor: colors.border, backgroundColor: colors.surface }]} onPress={() => {}}>
              <Text style={[styles.iconBtnText, { color: colors.textSecondary }]}>⟳</Text>
            </TouchableOpacity>
            <TouchableOpacity style={[styles.avatar, { backgroundColor: colors.primarySoft }]} onPress={() => navigation.navigate('Profile')}>
              {profileAvatar ? (
                <Image source={profileAvatar} style={styles.headerAvatarImage} />
              ) : (
                <Text style={[styles.avatarText, { color: colors.primary }]}>{profileInitials}</Text>
              )}
            </TouchableOpacity>
          </View>
        </View>

        <Card style={[styles.calorieCard, { borderColor: colors.primaryLight, overflow: 'hidden' }]}> 
          <View style={[styles.calorieOverlayShape, { backgroundColor: colors.primarySoft }]} />
          <View style={styles.calorieHeader}>
            <View>
              <Text style={[styles.calorieTitle, { color: colors.text }]}>Daily Calories</Text>
              <Text style={[styles.calorieSubtitle, { color: colors.textSecondary }]}>Your core energy target for today</Text>
              <Text style={[styles.calorieStreakText, { color: colors.textSecondary }]}>🔥 {xpProgression.streak} day streak</Text>
            </View>
            <View style={[styles.liveBadge, { backgroundColor: colors.primarySoft, borderColor: colors.primaryLight }]}>
              <Text style={[styles.liveBadgeText, { color: colors.primary }]}>Live</Text>
            </View>
          </View>

          <View style={styles.calorieRow}>
            <ProgressRing size={126} strokeWidth={14} progress={caloriesProgress} color={colors.calories} trackColor={colors.surfaceSecondary}>
              <Text style={[styles.calNumber, { color: colors.text }]}>{Math.round(consumed.calories)}</Text>
              <Text style={[styles.calLabel, { color: colors.textSecondary }]}>/ {Math.round(calorieGoal)} kcal</Text>
            </ProgressRing>

            <View style={styles.calorieInfo}>
              <Text style={[styles.remainingValue, { color: colors.text }]}>{Math.round(caloriesRemaining)} kcal</Text>
              <Text style={[styles.remainingSub, { color: colors.textSecondary }]}>remaining</Text>

              <View style={styles.badgeRow}>
                <View style={[styles.metaBadge, { backgroundColor: colors.surfaceSecondary }]}>
                  <Text style={[styles.metaBadgeText, { color: colors.textSecondary }]}>{usedPercent}% used</Text>
                </View>
                <View style={[styles.metaBadge, { backgroundColor: colors.surfaceSecondary }]}>
                  <Text style={[styles.metaBadgeText, { color: colors.textSecondary }]}>Goal {Math.round(calorieGoal)} kcal</Text>
                </View>
              </View>

              <TouchableOpacity style={[styles.editCalorieBtn, { backgroundColor: colors.primary }]} onPress={() => setIsCalorieModalOpen(true)}>
                <Text style={styles.editCalorieTitle}>Edit Calories</Text>
              </TouchableOpacity>
            </View>
          </View>
        </Card>

        <Card style={[styles.xpCard, { borderColor: colors.primaryLight, shadowColor: colors.primary, shadowOpacity: 0.15 }]}> 
          <TouchableOpacity onPress={toggleDailyXpExpanded} activeOpacity={0.9}>
            <View style={styles.xpHeaderRow}>
              <View>
                <Text style={[styles.sectionTitle, { color: colors.text }]}>🏆 Daily XP</Text>
                <Text style={[styles.sectionSub, { color: colors.textTertiary || colors.textSecondary }]}>Goal-aware nutrition progression</Text>
              </View>
              <View style={styles.xpHeaderRight}>
                <View style={[styles.xpBadge, { backgroundColor: colors.primarySoft }]}> 
                  <Text style={[styles.xpBadgeText, { color: colors.primary }]}>{Math.round(dailyXp?.totalXp || 0)} XP</Text>
                </View>
                <View style={styles.xpHeaderMetaRight}>
                  <Text style={[styles.xpDropdownArrow, { color: colors.textSecondary }]}>{isDailyXpExpanded ? '↑' : '↓'}</Text>
                  <Text style={[styles.xpLevelLabel, { color: colors.textSecondary }]}>Lv {xpProgression.level}</Text>
                </View>
              </View>
            </View>

            <View style={styles.xpCollapsedSummaryRow}>
              <Text style={[styles.xpCollapsedSummaryText, { color: colors.textTertiary || colors.textSecondary }]}>Nutrition {Math.round(dailyXp?.nutritionScore || 0)}% • Alignment {Math.round(dailyXp?.goalAlignmentScore || 0)}%</Text>
            </View>
          </TouchableOpacity>

          {isDailyXpExpanded ? (
            <>
              <View style={styles.xpStatsGrid}>
                <View style={[styles.xpStatItem, { backgroundColor: colors.surfaceSecondary }]}> 
                  <Text style={[styles.xpStatLabel, { color: colors.textSecondary }]}>Nutrition Score</Text>
                  <Text style={[styles.xpStatValue, { color: colors.text }]}>{Math.round(dailyXp?.nutritionScore || 0)}%</Text>
                </View>
                <View style={[styles.xpStatItem, { backgroundColor: colors.surfaceSecondary }]}> 
                  <Text style={[styles.xpStatLabel, { color: colors.textSecondary }]}>Goal Alignment</Text>
                  <Text style={[styles.xpStatValue, { color: colors.text }]}>{Math.round(dailyXp?.goalAlignmentScore || 0)}%</Text>
                </View>
                <View style={[styles.xpStatItem, { backgroundColor: colors.surfaceSecondary }]}> 
                  <Text style={[styles.xpStatLabel, { color: colors.textSecondary }]}>Level</Text>
                  <Text style={[styles.xpStatValue, { color: colors.text }]}>Lv {xpProgression.level}</Text>
                </View>
                <View style={[styles.xpStatItem, { backgroundColor: colors.surfaceSecondary }]}> 
                  <Text style={[styles.xpStatLabel, { color: colors.textSecondary }]}>Streak</Text>
                  <Text style={[styles.xpStatValue, { color: colors.text }]}>🔥 {xpProgression.streak}</Text>
                </View>
              </View>

              <View style={styles.xpProgressWrap}>
                <Text style={[styles.xpProgressLabel, { color: colors.textSecondary }]}>Level progress {Math.round(xpProgression.xpIntoLevel)} / 500 XP</Text>
                <ProgressBar
                  progress={(xpProgression.xpIntoLevel || 0) / 500}
                  color={colors.primary}
                  height={6}
                />
              </View>

              <View style={styles.xpBreakdownWrap}>
                {xpCategoryRows.map((row) => {
                  const earned = Number(dailyXp?.breakdown?.[row.key] || 0);
                  const max = Number(dailyXp?.maxByCategory?.[row.key] || 0);
                  return (
                    <View key={row.key} style={styles.xpBreakdownRow}>
                      <Text style={[styles.xpBreakdownLabel, { color: colors.textSecondary }]}>{row.label}</Text>
                      <Text style={[styles.xpBreakdownValue, { color: colors.text }]}>{Math.round(earned)} / {Math.round(max)} XP</Text>
                    </View>
                  );
                })}
              </View>

              <View style={[styles.xpReasonCard, { backgroundColor: colors.surfaceSecondary }]}> 
                <Text style={[styles.xpReasonTitle, { color: colors.text }]}>What was earned</Text>
                {xpReasonPreview.length === 0 ? (
                  <Text style={[styles.xpReasonText, { color: colors.textSecondary }]}>Log meals and hydration to start earning XP.</Text>
                ) : (
                  xpReasonPreview.map((reason, index) => {
                    const positive = reason.startsWith('+');
                    return (
                      <Text
                        key={`xp-reason-${index}`}
                        style={[
                          styles.xpReasonText,
                          { color: positive ? colors.success : colors.danger },
                        ]}
                      >
                        {reason}
                      </Text>
                    );
                  })
                )}
              </View>

              <View style={[styles.xpMissedCard, { backgroundColor: colors.surfaceSecondary }]}> 
                <Text style={[styles.xpReasonTitle, { color: colors.text }]}>What was missed / point loss</Text>
                {xpMissedRows.length === 0 ? (
                  <Text style={[styles.xpReasonText, { color: colors.success }]}>No major XP misses right now. Strong day.</Text>
                ) : (
                  xpMissedRows.map((row) => (
                    <Text key={`xp-missed-${row.key}`} style={[styles.xpReasonText, { color: colors.textSecondary }]}>
                      {row.label}: earned {Math.round(row.earned)} of {Math.round(row.max)} XP (missed {Math.round(row.missed)})
                    </Text>
                  ))
                )}
              </View>

              <Text style={[styles.xpFooterText, { color: colors.textSecondary }]}>Total XP: {Math.round(xpProgression.totalXp)} • Weekly XP: {Math.round(xpProgression.weeklyConsistencyXp)}</Text>
              {xpProgression.badges?.length > 0 && (
                <View style={styles.xpBadgeRow}>
                  {xpProgression.badges.map((badge) => (
                    <View key={badge} style={[styles.xpBadgeChip, { backgroundColor: colors.primarySoft }]}> 
                      <Text style={[styles.xpBadgeChipText, { color: colors.primary }]}>{badge}</Text>
                    </View>
                  ))}
                </View>
              )}
            </>
          ) : null}
        </Card>

        <View style={styles.sectionHeaderRow}>
          <View>
            <Text style={[styles.sectionTitle, { color: colors.text }]}>Macros</Text>
            <Text style={[styles.sectionSub, { color: colors.textSecondary }]}>Track today's macro progress</Text>
          </View>
          <TouchableOpacity style={[styles.editMacrosPill, { backgroundColor: colors.primarySoft, borderColor: colors.primaryLight }]} onPress={() => setIsMacroModalOpen(true)}>
            <Text style={[styles.editMacrosPillText, { color: colors.primary }]}>Edit Macros</Text>
          </TouchableOpacity>
        </View>

        <View style={styles.macroRow}>
          {macroData.map((macro) => (
            <Card key={macro.key} style={styles.macroCard}>
              <View style={styles.macroTopRow}>
                <Text style={[styles.macroLabel, { color: colors.textSecondary }]}>{macro.label}</Text>
                <Text style={[styles.macroPercent, { color: colors.primary }]}>{Math.round((macro.consumed / macro.target) * 100 || 0)}%</Text>
              </View>
              <Text style={[styles.macroValue, { color: colors.text }]}>
                {Math.round(macro.consumed)}/{Math.round(macro.target)}g
              </Text>
              <ProgressBar progress={(macro.consumed || 0) / (macro.target || 1)} color={macro.color} height={6} style={styles.macroBar} />
            </Card>
          ))}
        </View>

        <Card style={styles.waterCard}>
          <View style={styles.waterHeader}>
            <View>
              <Text style={[styles.sectionTitle, { color: colors.text }]}>💧 Hydration Tracker</Text>
              <Text style={[styles.sectionSub, { color: colors.textSecondary }]}>Track your daily water and drink intake</Text>
            </View>
          </View>

          <View style={styles.hydrationProgressSection}>
            <View style={styles.hydrationStats}>
              <Text style={[styles.hydrationValue, { color: colors.text }]}>
                {hydrationConsumedMl} ml
              </Text>
              <Text style={[styles.hydrationTarget, { color: colors.textSecondary }]}>
                / {hydrationTargetMl} ml
              </Text>
            </View>
            <View style={[styles.hydrationProgressTrack, { backgroundColor: colors.surfaceSecondary }]}> 
              <Animated.View style={[styles.hydrationProgressFill, { backgroundColor: colors.primary, width: hydrationProgressWidth }]} />
            </View>
            {hydrationMineralItems.length > 0 && (
              <View style={[styles.hydrationNutrientsCard, { backgroundColor: colors.surfaceSecondary, borderColor: colors.border }]}> 
                <Text style={[styles.hydrationNutrientsTitle, { color: colors.textSecondary }]}>Hydration Minerals (today)</Text>
                <View style={styles.hydrationNutrientsRow}>
                  {hydrationMineralItems.map((item) => (
                    <View key={item.key} style={[styles.hydrationNutrientChip, { backgroundColor: colors.primarySoft }]}> 
                      <Text style={[styles.hydrationNutrientLabel, { color: colors.primary }]}>{item.label}</Text>
                      <Text style={[styles.hydrationNutrientValue, { color: colors.text }]}>{Math.round(item.value)}mg</Text>
                    </View>
                  ))}
                </View>
              </View>
            )}
          </View>

          <View style={styles.drinkTypeSection}>
            <Text style={[styles.drinkTypeLabel, { color: colors.textSecondary }]}>Drink Type:</Text>
            <View style={styles.drinkTypeGrid}>
              {Object.entries(DRINK_TYPES).map(([key, drink]) => (
                <TouchableOpacity
                  key={key}
                  style={[
                    styles.drinkTypeBtn,
                    {
                      backgroundColor: selectedDrinkType === key ? colors.primary : 'transparent',
                      borderColor: selectedDrinkType === key ? colors.primary : colors.border,
                    },
                  ]}
                  onPress={() => setSelectedDrinkType(key)}
                  activeOpacity={0.8}
                >
                  <Text style={styles.drinkTypeEmoji}>{drink.emoji}</Text>
                  <Text
                    style={[
                      styles.drinkTypeName,
                      { color: selectedDrinkType === key ? colors.surface : colors.textSecondary },
                    ]}
                    numberOfLines={1}
                  >
                    {drink.name.split(' ')[0]}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>
          </View>

          <View style={styles.waterButtonGrid}>
            <TouchableOpacity style={[styles.waterBtn, { backgroundColor: colors.primarySoft, borderColor: colors.primary }]} onPress={() => addWater(250, 'ml', selectedDrinkType)} activeOpacity={0.8}>
              <Text style={[styles.waterBtnLabel, { color: colors.primary }]}>250ml</Text>
            </TouchableOpacity>
            <TouchableOpacity style={[styles.waterBtn, { backgroundColor: colors.primarySoft, borderColor: colors.primary }]} onPress={() => addWater(1, 'cup', selectedDrinkType)} activeOpacity={0.8}>
              <Text style={[styles.waterBtnLabel, { color: colors.primary }]}>1 Cup</Text>
            </TouchableOpacity>
            <TouchableOpacity style={[styles.waterBtn, { backgroundColor: colors.primarySoft, borderColor: colors.primary }]} onPress={() => addWater(8, 'oz', selectedDrinkType)} activeOpacity={0.8}>
              <Text style={[styles.waterBtnLabel, { color: colors.primary }]}>8 oz</Text>
            </TouchableOpacity>
            <TouchableOpacity style={[styles.waterBtn, { backgroundColor: colors.primarySoft, borderColor: colors.primary }]} onPress={() => addWater(500, 'ml', selectedDrinkType)} activeOpacity={0.8}>
              <Text style={[styles.waterBtnLabel, { color: colors.primary }]}>500ml</Text>
            </TouchableOpacity>
            <TouchableOpacity style={[styles.waterBtn, { backgroundColor: colors.primarySoft, borderColor: colors.primary }]} onPress={() => addWater(16, 'oz', selectedDrinkType)} activeOpacity={0.8}>
              <Text style={[styles.waterBtnLabel, { color: colors.primary }]}>16 oz</Text>
            </TouchableOpacity>
            <TouchableOpacity style={[styles.waterBtn, { backgroundColor: colors.primarySoft, borderColor: colors.primary }]} onPress={() => addWater(1, 'l', selectedDrinkType)} activeOpacity={0.8}>
              <Text style={[styles.waterBtnLabel, { color: colors.primary }]}>1L</Text>
            </TouchableOpacity>
          </View>

          <TouchableOpacity style={[styles.customWaterBtn, { backgroundColor: colors.primary }]} onPress={() => setIsWaterModalOpen(true)} activeOpacity={0.8}>
            <Text style={styles.customWaterBtnText}>+ Custom Amount</Text>
          </TouchableOpacity>
        </Card>

        <TouchableOpacity style={[styles.scanCta, { backgroundColor: colors.primary, shadowColor: colors.shadowColor, shadowOpacity: colors.shadowOpacity }]} onPress={() => setIsMealEntryModalOpen(true)} activeOpacity={0.9}>
          <View style={[styles.scanIconWrap, { backgroundColor: colors.primaryDark }]}>
            <Text style={styles.scanIcon}>🍽️</Text>
          </View>
          <View style={styles.scanCtaText}>
            <Text style={styles.scanTitle}>Scan or Log Meal</Text>
            <Text style={styles.scanSubtitle}>Photo, upload, or choose from saved meals</Text>
          </View>
          <Text style={styles.scanArrow}>→</Text>
        </TouchableOpacity>

        <Card style={styles.favoritesCard}>
          <View style={styles.sectionHeaderRow}>
            <View>
              <Text style={[styles.sectionTitle, { color: colors.text }]}>⭐ Your Favorites</Text>
              <Text style={[styles.sectionSub, { color: colors.textSecondary }]}>
                {favoriteMeals.length > 0 ? `${favoriteMeals.length} saved favorites` : 'Save meals to quickly log them later'}
              </Text>
            </View>
          </View>

          {favoriteMeals.length === 0 ? (
            <Text style={[styles.emptyFavoritesText, { color: colors.textSecondary }]}>Save meals from scan or meal details to quickly reuse them.</Text>
          ) : (
            <View style={styles.favoriteList}>
              {favoriteMeals.slice(0, 4).map((fav) => (
                <TouchableOpacity
                  key={fav.templateId}
                  style={[styles.favoriteItem, { backgroundColor: colors.surfaceSecondary, borderColor: colors.border }]}
                  onPress={() => handleSavedMealSelected(fav)}
                  activeOpacity={0.85}
                >
                  <View style={[styles.favoriteThumb, { backgroundColor: colors.primarySoft }]}> 
                    {fav.imageUri ? <Image source={{ uri: fav.imageUri }} style={styles.favoriteImage} resizeMode="cover" /> : <Text style={styles.favoriteEmoji}>⭐</Text>}
                  </View>
                  <View style={styles.favoriteInfo}>
                    <Text style={[styles.favoriteName, { color: colors.text }]} numberOfLines={1}>{fav.name}</Text>
                    <Text style={[styles.favoriteMeta, { color: colors.textSecondary }]}>{Math.round(fav.calories)} kcal</Text>
                  </View>
                  <TouchableOpacity
                    style={[styles.favoriteLogBtn, { backgroundColor: colors.primary }]}
                    onPress={() => handleSavedMealSelected(fav)}
                    activeOpacity={0.85}
                  >
                    <Text style={styles.favoriteLogBtnText}>Log</Text>
                  </TouchableOpacity>
                </TouchableOpacity>
              ))}
            </View>
          )}
        </Card>

        <Text style={[styles.sectionTitle, { color: colors.text }]}>Today's Meals</Text>
        <Text style={[styles.sectionSub, { color: colors.textSecondary }]}>{meals.length} meals logged</Text>

        {meals.length === 0 ? (
          <Card style={styles.emptyMealCard}>
            <View style={[styles.emptyMealIconWrap, { backgroundColor: colors.primarySoft }]}>
              <Text style={[styles.emptyMealIcon, { color: colors.primary }]}>📷</Text>
            </View>
            <Text style={[styles.emptyMealTitle, { color: colors.text }]}>No meals logged yet</Text>
            <Text style={[styles.emptyMealSub, { color: colors.textSecondary }]}>Start by scanning your first meal to build today's nutrition timeline.</Text>
            <TouchableOpacity
              style={[styles.emptyMealBtn, { backgroundColor: colors.primary }]}
              onPress={() => {
                setScanModalTab('scan');
                setIsScanModalOpen(true);
              }}
            >
              <Text style={styles.emptyMealBtnText}>Scan a Meal</Text>
            </TouchableOpacity>
          </Card>
        ) : (
          <View style={styles.mealList}>
            {meals.map((meal) => (
              <TouchableOpacity key={meal.id} style={[styles.mealCard, { backgroundColor: colors.surface, borderColor: colors.border, shadowColor: colors.shadowColor, shadowOpacity: colors.shadowOpacity }]} onPress={() => setSelectedMeal(meal)} activeOpacity={0.9}>
                <View style={[styles.mealThumb, { backgroundColor: colors.primarySoft }]}> 
                  {meal.imageUri ? <Image source={{ uri: meal.imageUri }} style={styles.mealImage} resizeMode="cover" /> : <Text style={styles.mealEmoji}>🍽️</Text>}
                </View>
                <View style={styles.mealInfo}>
                  <Text style={[styles.mealName, { color: colors.text }]}>{meal.name}</Text>
                  <Text style={[styles.mealMeta, { color: colors.textSecondary }]}>{toMealTimeLabel(meal.timestamp)}</Text>
                </View>
                <Text style={[styles.mealCalories, { color: colors.textSecondary }]}>{meal.calories} kcal</Text>
              </TouchableOpacity>
            ))}
          </View>
        )}

        <Card style={styles.insightsCard}>
          <TouchableOpacity style={styles.insightHeader} onPress={() => setIsInsightsExpanded((prev) => !prev)} activeOpacity={0.85}>
            <View style={styles.insightHeaderText}>
              <Text style={[styles.insightTitle, { color: colors.text }]}>Nutrition Insights</Text>
              <Text style={[styles.insightSub, { color: colors.textSecondary }]}>Daily vitamins, minerals, fats, and amino targets</Text>
              <Text style={[styles.insightHint, { color: colors.textSecondary }]}>Tap to view profile-based vitamins, minerals, amino acids, fatty acids, and custom targets.</Text>
              <View style={styles.insightSummaryRow}>
                <View style={[styles.summaryChip, { backgroundColor: colors.surfaceSecondary }]}>
                  <Text style={[styles.summaryChipText, { color: colors.textSecondary }]}>{nutrientSummaryCounts.low} low</Text>
                </View>
                <View style={[styles.summaryChip, { backgroundColor: colors.primarySoft }]}>
                  <Text style={[styles.summaryChipText, { color: colors.primary }]}>{nutrientSummaryCounts.onTrack} on track</Text>
                </View>
                <View style={[styles.summaryChip, { backgroundColor: colors.surfaceSecondary }]}>
                  <Text style={[styles.summaryChipText, { color: colors.textSecondary }]}>{nutrientSummaryCounts.custom} customized</Text>
                </View>
                <View style={[styles.summaryChip, { backgroundColor: colors.dangerSoft || colors.surfaceSecondary }]}>
                  <Text style={[styles.summaryChipText, { color: colors.danger }]}>{nutrientSummaryCounts.over} over</Text>
                </View>
              </View>
            </View>
            <Text style={[styles.chevron, { color: colors.textSecondary }]}>{isInsightsExpanded ? '⌃' : '⌄'}</Text>
          </TouchableOpacity>

          {isInsightsExpanded ? (
            <View style={styles.insightSections}>
              {insightsSummary.map((section) => (
                <View key={section.key} style={styles.insightSectionBlock}>
                  <View style={styles.insightSectionHead}>
                    <View>
                      <Text style={[styles.insightSectionTitle, { color: colors.text }]}>{section.title}</Text>
                      <Text style={[styles.insightSectionSub, { color: colors.textSecondary }]}>{section.description}</Text>
                    </View>
                    <TouchableOpacity onPress={() => setViewAllCategory(section.key)}>
                      <Text style={[styles.viewAllText, { color: colors.primary }]}>view all</Text>
                    </TouchableOpacity>
                  </View>
                  {renderNutrientCards(section.key, 3)}
                </View>
              ))}
            </View>
          ) : null}
        </Card>

        <Card style={styles.goalCard}>
          <Text style={[styles.goalTitle, { color: colors.text }]}>Your Goal</Text>
          <Text style={[styles.goalSubline, { color: colors.textSecondary }]}>Goal: {goalFocusLabel}</Text>

          <View style={styles.goalTagRow}>
            <View style={[styles.goalTag, { backgroundColor: colors.primarySoft }]}>
              <Text style={[styles.goalTagText, { color: colors.primary }]}>INTERPRETED: {profileSettings.interpretedGoal.toUpperCase()}</Text>
            </View>
            <View style={[styles.goalTag, { backgroundColor: colors.surfaceSecondary }]}>
              <Text style={[styles.goalTagText, { color: colors.textSecondary }]}>CONFIDENCE: {profileSettings.confidence.toUpperCase()}</Text>
            </View>
          </View>

          <View style={styles.goalMetricsGrid}>
            <View style={[styles.goalMetricCard, { backgroundColor: colors.surfaceSecondary }]}>
              <Text style={[styles.metricLabel, { color: colors.textSecondary }]}>Diet plan</Text>
              <Text style={[styles.metricValue, { color: colors.text }]}>{profileSettings.planLabel}</Text>
            </View>
            <View style={[styles.goalMetricCard, { backgroundColor: colors.surfaceSecondary }]}>
              <Text style={[styles.metricLabel, { color: colors.textSecondary }]}>Activity</Text>
              <Text style={[styles.metricValue, { color: colors.text }]}>{profileSettings.activityMultiplier.toFixed(2)}x</Text>
            </View>
            <View style={[styles.goalMetricCard, { backgroundColor: colors.surfaceSecondary }]}>
              <Text style={[styles.metricLabel, { color: colors.textSecondary }]}>BMR</Text>
              <Text style={[styles.metricValue, { color: colors.text }]}>{Math.round(bmr)} kcal/day</Text>
            </View>
            <View style={[styles.goalMetricCard, { backgroundColor: colors.surfaceSecondary }]}>
              <Text style={[styles.metricLabel, { color: colors.textSecondary }]}>TDEE</Text>
              <Text style={[styles.metricValue, { color: colors.text }]}>{Math.round(tdee)} kcal/day</Text>
            </View>
          </View>

          <Text style={[styles.goalFootnote, { color: colors.textSecondary }]}>Targets are dynamic from your age, gender, height, weight, activity, and goal. You can override any value.</Text>

          <TouchableOpacity style={[styles.goalEditBtn, { backgroundColor: colors.primarySoft, borderColor: colors.primaryLight }]} onPress={() => setIsCalorieModalOpen(true)}>
            <Text style={[styles.goalEditText, { color: colors.primary }]}>Edit Goal & Targets →</Text>
          </TouchableOpacity>
        </Card>
      </ScrollView>

      <Modal visible={isCalorieModalOpen} transparent animationType="slide" onRequestClose={() => setIsCalorieModalOpen(false)}>
        <View style={styles.overlay}>
          <View style={[styles.sheet, { backgroundColor: colors.surface }]}> 
            <Text style={[styles.sheetTitle, { color: colors.text }]}>Edit Daily Calories</Text>
            <Text style={[styles.sheetSub, { color: colors.textSecondary }]}>Recommended: {recommendedCalories} kcal/day</Text>

            <View style={styles.goalTypeRow}>
              {['lose', 'maintain', 'gain', 'muscle'].map((type) => (
                <TouchableOpacity
                  key={type}
                  style={[
                    styles.goalTypeChip,
                    {
                      borderColor: goalDraft === type ? colors.primary : colors.border,
                      backgroundColor: goalDraft === type ? colors.primarySoft : colors.surfaceSecondary,
                    },
                  ]}
                  onPress={() => setGoalDraft(type)}
                >
                  <Text style={[styles.goalTypeText, { color: goalDraft === type ? colors.primary : colors.textSecondary }]}>{type.toUpperCase()}</Text>
                </TouchableOpacity>
              ))}
            </View>

            <Text style={[styles.fieldLabel, { color: colors.textSecondary }]}>Activity multiplier</Text>
            <View style={styles.stepperRow}>
              <TouchableOpacity style={[styles.stepperBtn, { borderColor: colors.border }]} onPress={() => setActivityDraft((v) => Math.max(1.1, Number((v - 0.05).toFixed(2))))}>
                <Text style={[styles.stepperText, { color: colors.text }]}>-</Text>
              </TouchableOpacity>
              <Text style={[styles.stepperValue, { color: colors.text }]}>{activityDraft.toFixed(2)}x</Text>
              <TouchableOpacity style={[styles.stepperBtn, { borderColor: colors.border }]} onPress={() => setActivityDraft((v) => Math.min(2.4, Number((v + 0.05).toFixed(2))))}>
                <Text style={[styles.stepperText, { color: colors.text }]}>+</Text>
              </TouchableOpacity>
            </View>

            <View style={styles.stepperRow}>
              <TouchableOpacity style={[styles.stepperBtn, { borderColor: colors.border }]} onPress={() => setCustomCalories((prev) => Math.max(1200, (prev ?? recommendedDraftCalories) - 100))}>
                <Text style={[styles.stepperText, { color: colors.text }]}>-100</Text>
              </TouchableOpacity>
              <Text style={[styles.stepperValue, { color: colors.text }]}>{Math.round(customCalories ?? recommendedDraftCalories)} kcal</Text>
              <TouchableOpacity style={[styles.stepperBtn, { borderColor: colors.border }]} onPress={() => setCustomCalories((prev) => (prev ?? recommendedDraftCalories) + 100)}>
                <Text style={[styles.stepperText, { color: colors.text }]}>+100</Text>
              </TouchableOpacity>
            </View>

            {goalError ? <Text style={[styles.goalErrorText, { color: colors.danger }]}>{goalError}</Text> : null}

            <View style={styles.sheetActions}>
              <Button title="Use Recommended" size="sm" variant="outline" onPress={() => setCustomCalories(recommendedDraftCalories)} style={styles.sheetBtn} />
              <Button
                title={savingGoal ? 'Saving...' : 'Save'}
                size="sm"
                onPress={applyGoalAndActivity}
                disabled={savingGoal}
                style={styles.sheetBtn}
              />
            </View>
          </View>
        </View>
      </Modal>

      <Modal visible={isMacroModalOpen} transparent animationType="slide" onRequestClose={() => setIsMacroModalOpen(false)}>
        <View style={styles.overlay}>
          <View style={[styles.sheet, { backgroundColor: colors.surface }]}> 
            <Text style={[styles.sheetTitle, { color: colors.text }]}>Edit Macros</Text>
            <Text style={[styles.sheetSub, { color: colors.textSecondary }]}>Macros only: protein, carbs, fat, and water.</Text>

            <TouchableOpacity style={[styles.autoMacroBtn, { backgroundColor: colors.primarySoft, borderColor: colors.primaryLight }]} onPress={() => setMacroTargets(autoMacroTargets)}>
              <Text style={[styles.autoMacroText, { color: colors.primary }]}>Auto-Calculate Macros</Text>
            </TouchableOpacity>

            {[
              ['protein', 'Protein (g/day)', 5],
              ['carbs', 'Carbs (g/day)', 5],
              ['fat', 'Fat (g/day)', 5],
              ['water', 'Water (L/day)', 0.1],
            ].map(([field, label]) => (
              <View key={field} style={styles.macroEditorRow}>
                <Text style={[styles.fieldLabel, { color: colors.textSecondary }]}>{label}</Text>
                <View style={styles.inlineStepper}>
                  <TouchableOpacity style={[styles.inlineStepBtn, { borderColor: colors.border }]} onPress={() => changeMacro(field, -1)}>
                    <Text style={[styles.inlineStepText, { color: colors.text }]}>-{field === 'water' ? '0.1' : '5'}</Text>
                  </TouchableOpacity>
                  <View style={[styles.inlineValueBox, { borderColor: colors.border }]}>
                    <Text style={[styles.inlineValueText, { color: colors.text }]}>{macroTargets[field]}</Text>
                  </View>
                  <TouchableOpacity style={[styles.inlineStepBtn, { borderColor: colors.border }]} onPress={() => changeMacro(field, 1)}>
                    <Text style={[styles.inlineStepText, { color: colors.text }]}>+{field === 'water' ? '0.1' : '5'}</Text>
                  </TouchableOpacity>
                </View>
              </View>
            ))}

            <View style={styles.sheetActions}>
              <Button title="Cancel" size="sm" variant="outline" onPress={() => setIsMacroModalOpen(false)} style={styles.sheetBtn} />
              <Button title="Save Macros" size="sm" onPress={() => setIsMacroModalOpen(false)} style={styles.sheetBtn} />
            </View>
          </View>
        </View>
      </Modal>

      <Modal visible={isScanModalOpen} transparent animationType="fade" onRequestClose={() => setIsScanModalOpen(false)}>
        <Pressable style={styles.overlay} onPress={() => setIsScanModalOpen(false)}>
          <Pressable style={[styles.scanSheet, { backgroundColor: colors.surface }]} onPress={() => {}}>
            <Text style={[styles.sheetTitle, { color: colors.text }]}>Scan Your Meal</Text>
            <Text style={[styles.sheetSub, { color: colors.textSecondary }]}>Scan, upload, or quickly log from favorites.</Text>

            <View style={styles.scanTabsRow}>
              {[
                { key: 'scan', label: 'Scan' },
                { key: 'upload', label: 'Upload' },
                { key: 'favorites', label: '⭐ Favorites' },
              ].map((tab) => (
                <TouchableOpacity
                  key={tab.key}
                  style={[
                    styles.scanTabBtn,
                    {
                      backgroundColor: scanModalTab === tab.key ? colors.primarySoft : colors.surfaceSecondary,
                      borderColor: scanModalTab === tab.key ? colors.primary : colors.border,
                    },
                  ]}
                  onPress={() => setScanModalTab(tab.key)}
                  activeOpacity={0.85}
                >
                  <Text style={[styles.scanTabText, { color: scanModalTab === tab.key ? colors.primary : colors.textSecondary }]}>{tab.label}</Text>
                </TouchableOpacity>
              ))}
            </View>

            {scanModalTab === 'scan' && (
              <View style={styles.sampleMealList}>
                {SAMPLE_MEALS.map((meal) => (
                  <View key={meal.name} style={[styles.sampleMealItem, { borderColor: colors.border }]}> 
                    <TouchableOpacity style={styles.sampleMealMain} onPress={() => addMeal(meal)} activeOpacity={0.85}>
                      <Text style={[styles.sampleMealName, { color: colors.text }]}>{meal.name}</Text>
                      <Text style={[styles.sampleMealMeta, { color: colors.textSecondary }]}>{meal.calories} kcal</Text>
                    </TouchableOpacity>
                    <TouchableOpacity
                      style={[styles.sampleMealSaveBtn, { backgroundColor: colors.primarySoft, borderColor: colors.primary }]}
                      onPress={() => handleSaveSampleMealToFavorites(meal)}
                      activeOpacity={0.85}
                    >
                      <Text style={[styles.sampleMealSaveText, { color: colors.primary }]}>⭐ Save</Text>
                    </TouchableOpacity>
                  </View>
                ))}
              </View>
            )}

            {scanModalTab === 'upload' && (
              <Button title="Upload Meal Photo" size="sm" onPress={handlePickMealImage} style={styles.fullWidth} />
            )}

            {scanModalTab === 'favorites' && (
              <View style={styles.sampleMealList}>
                {favoriteMeals.length === 0 ? (
                  <Text style={[styles.emptyFavoritesText, { color: colors.textSecondary }]}>Save meals to quickly log them later.</Text>
                ) : (
                  favoriteMeals.map((meal) => (
                    <TouchableOpacity
                      key={meal.templateId}
                      style={[styles.sampleMealItem, { borderColor: colors.border }]}
                      onPress={() => {
                        setIsScanModalOpen(false);
                        handleSavedMealSelected(meal);
                      }}
                      activeOpacity={0.85}
                    >
                      <Text style={[styles.sampleMealName, { color: colors.text }]}>{meal.name}</Text>
                      <Text style={[styles.sampleMealMeta, { color: colors.textSecondary }]}>{Math.round(meal.calories)} kcal</Text>
                    </TouchableOpacity>
                  ))
                )}
              </View>
            )}
          </Pressable>
        </Pressable>
      </Modal>

      <Modal visible={Boolean(selectedMeal)} transparent animationType="slide" onRequestClose={() => setSelectedMeal(null)}>
        <View style={styles.overlay}>
          <View style={[styles.sheet, { backgroundColor: colors.surface }]}> 
            {selectedMeal ? (
              <>
                <Text style={[styles.sheetTitle, { color: colors.text }]}>{selectedMeal.name}</Text>
                <Text style={[styles.sheetSub, { color: colors.textSecondary }]}>{toMealTimeLabel(selectedMeal.timestamp)}</Text>
                <View style={styles.mealDetailRow}>
                  <Text style={[styles.detailLabel, { color: colors.textSecondary }]}>Calories</Text>
                  <Text style={[styles.detailValue, { color: colors.text }]}>{selectedMeal.calories} kcal</Text>
                </View>
                <View style={styles.mealDetailRow}>
                  <Text style={[styles.detailLabel, { color: colors.textSecondary }]}>Protein</Text>
                  <Text style={[styles.detailValue, { color: colors.text }]}>{selectedMeal.protein} g</Text>
                </View>
                <View style={styles.mealDetailRow}>
                  <Text style={[styles.detailLabel, { color: colors.textSecondary }]}>Carbs</Text>
                  <Text style={[styles.detailValue, { color: colors.text }]}>{selectedMeal.carbs} g</Text>
                </View>
                <View style={styles.mealDetailRow}>
                  <Text style={[styles.detailLabel, { color: colors.textSecondary }]}>Fat</Text>
                  <Text style={[styles.detailValue, { color: colors.text }]}>{selectedMeal.fat} g</Text>
                </View>

                <TouchableOpacity
                  style={[
                    styles.favoriteToggleBtn,
                    {
                      backgroundColor: selectedMealIsFavorite ? colors.warning : colors.primarySoft,
                      borderColor: selectedMealIsFavorite ? colors.warning : colors.primary,
                    },
                  ]}
                  onPress={handleToggleFavoriteFromSelectedMeal}
                  activeOpacity={0.85}
                >
                  <Text style={[styles.favoriteToggleBtnText, { color: selectedMealIsFavorite ? '#ffffff' : colors.primary }]}> 
                    {selectedMealIsFavorite ? '⭐ Remove from Favorites' : '☆ Save to Favorites'}
                  </Text>
                </TouchableOpacity>

                <View style={styles.sheetActions}>
                  <Button title="Close" size="sm" variant="outline" onPress={() => setSelectedMeal(null)} style={styles.sheetBtn} />
                  <Button title="Delete Meal" size="sm" onPress={() => removeMeal(selectedMeal.id)} style={styles.sheetBtn} />
                </View>
              </>
            ) : null}
          </View>
        </View>
      </Modal>

      <Modal visible={Boolean(viewAllCategory)} transparent animationType="slide" onRequestClose={() => setViewAllCategory(null)}>
        <View style={styles.overlay}>
          <View style={[styles.largeSheet, { backgroundColor: colors.surface, paddingBottom: insets.bottom + 8 }]}> 
            {viewAllCategory ? (
              <>
                <View style={styles.fullModalHeader}>
                  <View>
                    <Text style={[styles.sheetTitle, { color: colors.text }]}>{CATEGORY_TITLES[viewAllCategory]}</Text>
                    <Text style={[styles.sheetSub, { color: colors.textSecondary }]}>Tap info for why each nutrient matters. Edit targets and save instantly.</Text>
                  </View>
                  <TouchableOpacity style={[styles.closeCircle, { borderColor: colors.border, backgroundColor: colors.surfaceSecondary }]} onPress={() => setViewAllCategory(null)}>
                    <Text style={[styles.closeCircleText, { color: colors.textSecondary }]}>×</Text>
                  </TouchableOpacity>
                </View>

                <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={styles.largeSheetBody}>
                  {renderNutrientCards(viewAllCategory, null)}
                </ScrollView>

                <View style={[styles.sheetActions, styles.stickyActions]}>
                  <Button
                    title="Edit Targets"
                    size="sm"
                    variant="outline"
                    onPress={() => {
                      setEditingCategoryTargets(viewAllCategory);
                      setViewAllCategory(null);
                    }}
                    style={styles.sheetBtn}
                  />
                  <Button title="Close" size="sm" onPress={() => setViewAllCategory(null)} style={styles.sheetBtn} />
                </View>
              </>
            ) : null}
          </View>
        </View>
      </Modal>

      <Modal visible={Boolean(editingNutrient)} transparent animationType="fade" onRequestClose={() => setEditingNutrient(null)}>
        <Pressable style={styles.overlay} onPress={() => setEditingNutrient(null)}>
          <Pressable style={[styles.smallSheet, { backgroundColor: colors.surface }]} onPress={() => {}}>
            <Text style={[styles.sheetTitle, { color: colors.text }]}>Edit {editingNutrient?.label} Target</Text>
            <Text style={[styles.sheetSub, { color: colors.textSecondary }]}>Override target value</Text>
            <TextInput
              style={[styles.targetInput, { borderColor: colors.border, color: colors.text, backgroundColor: colors.surfaceSecondary }]}
              value={nutrientEditValue}
              onChangeText={setNutrientEditValue}
              keyboardType="decimal-pad"
            />
            {targetError ? <Text style={[styles.goalErrorText, { color: colors.danger }]}>{targetError}</Text> : null}
            <View style={styles.sheetActions}>
              <Button title="Cancel" size="sm" variant="outline" onPress={() => setEditingNutrient(null)} style={styles.sheetBtn} />
              <Button title="Save" size="sm" onPress={saveNutrientEdit} style={styles.sheetBtn} />
            </View>
          </Pressable>
        </Pressable>
      </Modal>

      <Modal visible={Boolean(editingCategoryTargets)} transparent animationType="slide" onRequestClose={() => setEditingCategoryTargets(null)}>
        <View style={styles.overlay}>
          <View style={[styles.largeSheet, { backgroundColor: colors.surface, paddingBottom: insets.bottom + 8 }]}> 
            {editingCategoryTargets ? (
              <>
                <View style={styles.fullModalHeader}>
                  <View>
                    <Text style={[styles.sheetTitle, { color: colors.text }]}>Edit {CATEGORY_TITLES[editingCategoryTargets]} Targets</Text>
                    <Text style={[styles.sheetSub, { color: colors.textSecondary }]}>Set custom values or reset to recommended.</Text>
                  </View>
                  <TouchableOpacity style={[styles.closeCircle, { borderColor: colors.border, backgroundColor: colors.surfaceSecondary }]} onPress={() => setEditingCategoryTargets(null)}>
                    <Text style={[styles.closeCircleText, { color: colors.textSecondary }]}>×</Text>
                  </TouchableOpacity>
                </View>

                <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={styles.largeSheetBody}>
                  {(NUTRIENT_TEMPLATE[editingCategoryTargets] || []).map((item) => {
                    const currentTarget = Number(nutrientTargets[item.key] ?? item.target);
                    const mode = nutrientTargetModes[item.key] || 'auto';
                    return (
                      <View key={item.key} style={[styles.editTargetRow, { backgroundColor: colors.surfaceSecondary }]}> 
                        <View style={styles.editTargetHeader}>
                          <Text style={[styles.editTargetTitle, { color: colors.text }]}>{item.label}</Text>
                          <Text style={[styles.editTargetMode, { color: mode === 'custom' ? colors.primary : colors.textSecondary }]}>{mode.toUpperCase()}</Text>
                        </View>
                        <View style={styles.inlineStepper}>
                          <TouchableOpacity
                            style={[styles.inlineStepBtn, { borderColor: colors.border }]}
                            onPress={() => {
                              const next = Math.max(0.1, Number((currentTarget - (item.unit === 'g' ? 0.1 : 1)).toFixed(2)));
                              setNutrientTargets((prev) => ({ ...prev, [item.key]: next }));
                              setNutrientTargetModes((prev) => ({ ...prev, [item.key]: 'custom' }));
                            }}
                          >
                            <Text style={[styles.inlineStepText, { color: colors.text }]}>-</Text>
                          </TouchableOpacity>
                          <View style={[styles.inlineValueBox, { borderColor: colors.border }]}>
                            <Text style={[styles.inlineValueText, { color: colors.text }]}>{formatAmount(currentTarget)} {item.unit.toUpperCase()}</Text>
                          </View>
                          <TouchableOpacity
                            style={[styles.inlineStepBtn, { borderColor: colors.border }]}
                            onPress={() => {
                              const next = Number((currentTarget + (item.unit === 'g' ? 0.1 : 1)).toFixed(2));
                              setNutrientTargets((prev) => ({ ...prev, [item.key]: next }));
                              setNutrientTargetModes((prev) => ({ ...prev, [item.key]: 'custom' }));
                            }}
                          >
                            <Text style={[styles.inlineStepText, { color: colors.text }]}>+</Text>
                          </TouchableOpacity>
                        </View>
                      </View>
                    );
                  })}
                </ScrollView>

                <View style={[styles.sheetActions, styles.stickyActions]}>
                  <Button title="Reset Section" size="sm" variant="outline" onPress={() => resetCategoryTargetsToAuto(editingCategoryTargets)} style={styles.sheetBtn} />
                  <Button
                    title="Save"
                    size="sm"
                    onPress={async () => {
                      try {
                        await updateProfile({ nutrientTargets, nutrientTargetModes });
                        setEditingCategoryTargets(null);
                      } catch (error) {
                        setTargetError(error?.message || 'Unable to save nutrient targets.');
                      }
                    }}
                    style={styles.sheetBtn}
                  />
                </View>
              </>
            ) : null}
          </View>
        </View>
      </Modal>

      <Modal visible={Boolean(infoNutrient)} transparent animationType="fade" onRequestClose={() => setInfoNutrient(null)}>
        <Pressable style={styles.overlay} onPress={() => setInfoNutrient(null)}>
          <Pressable style={[styles.smallSheet, { backgroundColor: colors.surface }]} onPress={() => {}}>
            <Text style={[styles.sheetTitle, { color: colors.text }]}>{infoNutrient?.label}</Text>
            <Text style={[styles.sheetSub, { color: colors.textSecondary }]}>Why it matters</Text>
            <Text style={[styles.infoBodyText, { color: colors.text }]}>
              {(infoNutrient && NUTRIENT_INFO[infoNutrient.key]?.purpose) || 'Supports key nutrition targets for your daily plan.'}
            </Text>
            <Text style={[styles.sheetSub, { color: colors.textSecondary, marginTop: 10 }]}>Common sources</Text>
            <Text style={[styles.infoBodyText, { color: colors.text }]}>
              {(infoNutrient && NUTRIENT_INFO[infoNutrient.key]?.sources) || 'Whole foods, balanced meals, and nutrient-dense choices.'}
            </Text>
            <View style={styles.sheetActions}>
              <Button title="Close" size="sm" onPress={() => setInfoNutrient(null)} style={styles.sheetBtn} />
            </View>
          </Pressable>
        </Pressable>
      </Modal>

      <Modal visible={isWaterModalOpen} transparent animationType="slide" onRequestClose={() => setIsWaterModalOpen(false)}>
        <View style={styles.overlay}>
          <View style={[styles.sheet, { backgroundColor: colors.surface }]}>
            <Text style={[styles.sheetTitle, { color: colors.text }]}>💧 Add Custom Hydration</Text>
            <Text style={[styles.sheetSub, { color: colors.textSecondary }]}>Enter amount and select drink type</Text>

            <View style={styles.hydrationInputRow}>
              <TextInput
                style={[styles.hydrationAmountInput, { borderColor: colors.border, color: colors.text, backgroundColor: colors.surfaceSecondary }]}
                placeholder="Amount"
                placeholderTextColor={colors.textSecondary}
                value={customWaterAmount}
                onChangeText={setCustomWaterAmount}
                keyboardType="decimal-pad"
              />

              <View style={[styles.hydrationUnitPicker, { backgroundColor: colors.surfaceSecondary, borderColor: colors.border }]}>
                <ScrollView horizontal showsHorizontalScrollIndicator={false}>
                  {['ml', 'l', 'oz', 'cup'].map((unit) => (
                    <TouchableOpacity
                      key={unit}
                      style={[
                        styles.unitOption,
                        {
                          backgroundColor: customWaterUnit === unit ? colors.primary : 'transparent',
                        },
                      ]}
                      onPress={() => setCustomWaterUnit(unit)}
                    >
                      <Text
                        style={[
                          styles.unitOptionText,
                          {
                            color: customWaterUnit === unit ? colors.surface : colors.text,
                            fontWeight: customWaterUnit === unit ? '700' : '400',
                          },
                        ]}
                      >
                        {unit.toUpperCase()}
                      </Text>
                    </TouchableOpacity>
                  ))}
                </ScrollView>
              </View>
            </View>

            <View style={styles.drinkTypeModalSection}>
              <Text style={[styles.drinkTypeModalLabel, { color: colors.text }]}>Select Drink Type:</Text>
              <ScrollView showsVerticalScrollIndicator={false} style={styles.drinkTypeModalGrid}>
                {Object.entries(DRINK_TYPES).map(([key, drink]) => (
                  <TouchableOpacity
                    key={key}
                    style={[
                      styles.drinkTypeModalOption,
                      {
                        backgroundColor: selectedDrinkType === key ? colors.primarySoft : colors.surfaceSecondary,
                        borderColor: selectedDrinkType === key ? colors.primary : colors.border,
                      },
                    ]}
                    onPress={() => setSelectedDrinkType(key)}
                  >
                    <Text style={styles.drinkTypeModalEmoji}>{drink.emoji}</Text>
                    <View>
                      <Text style={[styles.drinkTypeModalName, { color: colors.text }]}>
                        {drink.name}
                      </Text>
                      {drink.nutrients && Object.keys(drink.nutrients).length > 0 && (
                        <Text style={[styles.drinkTypeModalNutrients, { color: colors.textSecondary }]}>
                          With minerals & electrolytes
                        </Text>
                      )}
                    </View>
                    {selectedDrinkType === key && (
                      <Text style={{ fontSize: 18, marginLeft: 'auto' }}>✓</Text>
                    )}
                  </TouchableOpacity>
                ))}
              </ScrollView>
            </View>

            <View style={styles.waterUnitHint}>
              <Text style={[styles.waterUnitText, { color: colors.textSecondary }]}> 
                1 cup = 236.59ml | 8oz = 236.59ml | 16oz = 473.18ml | 1L = 1000ml
              </Text>
            </View>

            <View style={styles.sheetActions}>
              <Button
                title="Cancel"
                size="sm"
                variant="outline"
                onPress={() => {
                  setIsWaterModalOpen(false);
                  setCustomWaterAmount('');
                  setCustomWaterUnit('ml');
                }}
                style={styles.sheetBtn}
              />
              <Button
                title="Add Hydration"
                size="sm"
                onPress={() => addWater(customWaterAmount, customWaterUnit, selectedDrinkType)}
                style={styles.sheetBtn}
              />
            </View>
          </View>
        </View>
      </Modal>

      {/* Meal Entry Options Modal */}
      <Modal visible={isMealEntryModalOpen} transparent animationType="slide" onRequestClose={() => setIsMealEntryModalOpen(false)}>
        <View style={styles.overlay}>
          <View style={[styles.sheet, { backgroundColor: colors.surface }]}>
            <View style={styles.fullModalHeader}>
              <View>
                <Text style={[styles.sheetTitle, { color: colors.text }]}>Add Meal</Text>
                <Text style={[styles.sheetSub, { color: colors.textSecondary }]}>Scan, upload, or use saved meals</Text>
              </View>
              <TouchableOpacity style={[styles.closeCircle, { borderColor: colors.border, backgroundColor: colors.surfaceSecondary }]} onPress={() => setIsMealEntryModalOpen(false)}>
                <Text style={[styles.closeCircleText, { color: colors.textSecondary }]}>×</Text>
              </TouchableOpacity>
            </View>

            <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={styles.mealEntryOptions}>
              <TouchableOpacity
                style={[styles.mealEntryOption, { backgroundColor: colors.primarySoft, borderColor: colors.primary }]}
                onPress={() => {
                  setIsMealEntryModalOpen(false);
                  setScanModalTab('scan');
                  setIsScanModalOpen(true);
                }}
                activeOpacity={0.8}
              >
                <Text style={[styles.mealEntryOptionIcon, { color: colors.primary }]}>📷</Text>
                <View style={styles.mealEntryOptionText}>
                  <Text style={[styles.mealEntryOptionTitle, { color: colors.primary }]}>Take Photo</Text>
                  <Text style={[styles.mealEntryOptionSub, { color: colors.textSecondary }]}>Use camera to scan meal</Text>
                </View>
              </TouchableOpacity>

              <TouchableOpacity
                style={[styles.mealEntryOption, { backgroundColor: colors.primarySoft, borderColor: colors.primary }]}
                onPress={() => {
                  setIsMealEntryModalOpen(false);
                  handlePickMealImage();
                }}
                activeOpacity={0.8}
              >
                <Text style={[styles.mealEntryOptionIcon, { color: colors.primary }]}>🖼️</Text>
                <View style={styles.mealEntryOptionText}>
                  <Text style={[styles.mealEntryOptionTitle, { color: colors.primary }]}>Upload Photo</Text>
                  <Text style={[styles.mealEntryOptionSub, { color: colors.textSecondary }]}>Upload from gallery</Text>
                </View>
              </TouchableOpacity>

              {savedMeals.length > 0 && (
                <>
                  <TouchableOpacity
                    style={[styles.mealEntryOption, { backgroundColor: colors.surfaceSecondary, borderColor: colors.border }]}
                    onPress={() => {
                      setSavedMealBrowserMode('saved');
                      setIsMealEntryModalOpen(false);
                      setIsSavedMealsBrowserOpen(true);
                    }}
                    activeOpacity={0.8}
                  >
                    <Text style={[styles.mealEntryOptionIcon, { color: colors.textSecondary }]}>💾</Text>
                    <View style={styles.mealEntryOptionText}>
                      <Text style={[styles.mealEntryOptionTitle, { color: colors.text }]}>Saved Meals</Text>
                      <Text style={[styles.mealEntryOptionSub, { color: colors.textSecondary }]}>({savedMeals.length} templates)</Text>
                    </View>
                  </TouchableOpacity>

                  {favoriteMeals.length > 0 && (
                    <TouchableOpacity
                      style={[styles.mealEntryOption, { backgroundColor: colors.surfaceSecondary, borderColor: colors.border }]}
                      onPress={() => {
                        setSavedMealBrowserMode('favorites');
                        setIsMealEntryModalOpen(false);
                        setIsSavedMealsBrowserOpen(true);
                      }}
                      activeOpacity={0.8}
                    >
                      <Text style={[styles.mealEntryOptionIcon, { color: colors.textSecondary }]}>⭐</Text>
                      <View style={styles.mealEntryOptionText}>
                        <Text style={[styles.mealEntryOptionTitle, { color: colors.text }]}>Favorites</Text>
                        <Text style={[styles.mealEntryOptionSub, { color: colors.textSecondary }]}>({favoriteMeals.length} favorites)</Text>
                      </View>
                    </TouchableOpacity>
                  )}

                  {getRecentMeals().length > 0 && (
                    <TouchableOpacity
                      style={[styles.mealEntryOption, { backgroundColor: colors.surfaceSecondary, borderColor: colors.border }]}
                      onPress={() => {
                        setSavedMealBrowserMode('recent');
                        setIsMealEntryModalOpen(false);
                        setIsSavedMealsBrowserOpen(true);
                      }}
                      activeOpacity={0.8}
                    >
                      <Text style={[styles.mealEntryOptionIcon, { color: colors.textSecondary }]}>🕐</Text>
                      <View style={styles.mealEntryOptionText}>
                        <Text style={[styles.mealEntryOptionTitle, { color: colors.text }]}>Recent Meals</Text>
                        <Text style={[styles.mealEntryOptionSub, { color: colors.textSecondary }]}>Quick access to recent</Text>
                      </View>
                    </TouchableOpacity>
                  )}
                </>
              )}
            </ScrollView>

            <View style={styles.sheetActions}>
              <Button title="Close" size="sm" onPress={() => setIsMealEntryModalOpen(false)} style={styles.sheetBtn} />
            </View>
          </View>
        </View>
      </Modal>

      {/* Saved Meals Browser Modal */}
      <Modal visible={isSavedMealsBrowserOpen} transparent animationType="slide" onRequestClose={() => setIsSavedMealsBrowserOpen(false)}>
        <View style={styles.overlay}>
          <View style={[styles.largeSheet, { backgroundColor: colors.surface, paddingBottom: insets.bottom + 8 }]}>
            <View style={styles.fullModalHeader}>
              <View>
                <Text style={[styles.sheetTitle, { color: colors.text }]}>
                  {savedMealBrowserMode === 'saved' ? '💾 Saved Meals' : savedMealBrowserMode === 'favorites' ? '⭐ Favorites' : '🕐 Recent Meals'}
                </Text>
                <Text style={[styles.sheetSub, { color: colors.textSecondary }]}>Tap a meal to log with portion adjustment</Text>
              </View>
              <TouchableOpacity style={[styles.closeCircle, { borderColor: colors.border, backgroundColor: colors.surfaceSecondary }]} onPress={() => setIsSavedMealsBrowserOpen(false)}>
                <Text style={[styles.closeCircleText, { color: colors.textSecondary }]}>×</Text>
              </TouchableOpacity>
            </View>

            <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={styles.savedMealsList}>
              {(savedMealBrowserMode === 'saved'
                ? savedMeals
                : savedMealBrowserMode === 'favorites'
                  ? favoriteMeals
                  : getRecentMeals()
              ).map((savedMeal) => (
                <View key={savedMeal.templateId} style={[styles.savedMealItem, { backgroundColor: colors.surfaceSecondary }]}>
                  {savedMeal.imageUri && (
                    <Image source={{ uri: savedMeal.imageUri }} style={styles.savedMealImage} resizeMode="cover" />
                  )}
                  <View style={styles.savedMealInfo}>
                    <Text style={[styles.savedMealName, { color: colors.text }]}>{savedMeal.name}</Text>
                    <Text style={[styles.savedMealMeta, { color: colors.textSecondary }]}>
                      {Math.round(savedMeal.calories)} cal | {Math.round(savedMeal.protein)}g P {Math.round(savedMeal.carbs)}g C {Math.round(savedMeal.fat)}g F
                    </Text>
                    {savedMeal.lastUsedAt && (
                      <Text style={[styles.savedMealMeta, { color: colors.textSecondary, fontSize: 11 }]}>
                        Last used: {new Date(savedMeal.lastUsedAt).toLocaleDateString()}
                      </Text>
                    )}
                  </View>
                  <View style={styles.savedMealActions}>
                    <TouchableOpacity
                      style={[styles.savedMealActionBtn, { backgroundColor: colors.primary }]}
                      onPress={() => handleSavedMealSelected(savedMeal)}
                      activeOpacity={0.8}
                    >
                      <Text style={styles.savedMealActionBtnText}>Log</Text>
                    </TouchableOpacity>
                    {savedMealBrowserMode === 'saved' && (
                      <TouchableOpacity
                        style={[styles.savedMealActionBtn, { backgroundColor: favoriteMealIds.has(savedMeal.templateId) ? colors.warning : colors.surfaceSecondary, borderColor: colors.border, borderWidth: 1 }]}
                        onPress={() => handleToggleFavorite(savedMeal.templateId)}
                        activeOpacity={0.8}
                      >
                        <Text style={[styles.savedMealActionBtnText, { color: favoriteMealIds.has(savedMeal.templateId) ? '#fff' : colors.textSecondary }]}>
                          {favoriteMealIds.has(savedMeal.templateId) ? '⭐' : '☆'}
                        </Text>
                      </TouchableOpacity>
                    )}
                    <TouchableOpacity
                      style={[styles.savedMealActionBtn, { backgroundColor: colors.danger }]}
                      onPress={() => {
                        deleteSavedMeal(savedMeal.templateId);
                        Alert.alert('Deleted', `${savedMeal.name} removed`);
                      }}
                      activeOpacity={0.8}
                    >
                      <Text style={styles.savedMealActionBtnText}>×</Text>
                    </TouchableOpacity>
                  </View>
                </View>
              ))}
            </ScrollView>

            <View style={[styles.sheetActions, styles.stickyActions]}>
              <Button title="Back to Options" size="sm" variant="outline" onPress={() => { setIsSavedMealsBrowserOpen(false); setIsMealEntryModalOpen(true); }} style={styles.sheetBtn} />
              <Button title="Close" size="sm" onPress={() => setIsSavedMealsBrowserOpen(false)} style={styles.sheetBtn} />
            </View>
          </View>
        </View>
      </Modal>

      {/* Portion Adjustment Modal */}
      <Modal visible={isPortionAdjustOpen} transparent animationType="slide" onRequestClose={() => setIsPortionAdjustOpen(false)}>
        <View style={styles.overlay}>
          <View style={[styles.sheet, { backgroundColor: colors.surface }]}>
            <Text style={[styles.sheetTitle, { color: colors.text }]}>🍽️ Adjust Portion</Text>
            <Text style={[styles.sheetSub, { color: colors.textSecondary }]}>{selectedSavedMealForLog?.name}</Text>

            <View style={[styles.portionCard, { backgroundColor: colors.surfaceSecondary }]}>
              <Text style={[styles.portionLabel, { color: colors.textSecondary }]}>Serving Amount</Text>
              <Text style={[styles.portionValue, { color: colors.primary }]}>{effectivePortionMultiplier.toFixed(2)}x</Text>
              <View style={styles.portionUnitTabs}>
                {['x', 'g', 'oz'].map((unit) => (
                  <TouchableOpacity
                    key={unit}
                    style={[
                      styles.portionUnitBtn,
                      {
                        backgroundColor: portionInputUnit === unit ? colors.primarySoft : colors.surface,
                        borderColor: portionInputUnit === unit ? colors.primary : colors.border,
                      },
                    ]}
                    onPress={() => {
                      setPortionInputUnit(unit);
                      if (unit === 'x') setPortionInputValue(String(portionMultiplier));
                    }}
                  >
                    <Text style={[styles.portionUnitBtnText, { color: portionInputUnit === unit ? colors.primary : colors.textSecondary }]}>{unit === 'x' ? 'Multiplier' : unit.toUpperCase()}</Text>
                  </TouchableOpacity>
                ))}
              </View>
              <TextInput
                style={[styles.portionInput, { borderColor: colors.border, color: colors.text, backgroundColor: colors.surface }]}
                value={portionInputValue}
                onChangeText={setPortionInputValue}
                keyboardType="decimal-pad"
                placeholder={portionInputUnit === 'x' ? '1.0' : portionInputUnit === 'g' ? '100' : '3.5'}
                placeholderTextColor={colors.textSecondary}
              />
            </View>

            <View style={styles.portionControlRow}>
              <TouchableOpacity
                style={[styles.portionBtn, { backgroundColor: colors.primarySoft, borderColor: colors.primary }]}
                onPress={() => {
                  const next = Number(Math.max(0.25, portionMultiplier - 0.25).toFixed(2));
                  setPortionMultiplier(next);
                  setPortionInputUnit('x');
                  setPortionInputValue(String(next));
                }}
                activeOpacity={0.8}
              >
                <Text style={[styles.portionBtnText, { color: colors.primary }]}>− 0.25x</Text>
              </TouchableOpacity>

              <TouchableOpacity
                style={[styles.portionBtn, { backgroundColor: colors.surfaceSecondary, borderColor: colors.border }]}
                onPress={() => {
                  setPortionMultiplier(0.5);
                  setPortionInputUnit('x');
                  setPortionInputValue('0.5');
                }}
                activeOpacity={0.8}
              >
                <Text style={[styles.portionBtnText, { color: colors.text }]}>1/2</Text>
              </TouchableOpacity>

              <TouchableOpacity
                style={[styles.portionBtn, { backgroundColor: colors.surfaceSecondary, borderColor: colors.border }]}
                onPress={() => {
                  setPortionMultiplier(1);
                  setPortionInputUnit('x');
                  setPortionInputValue('1');
                }}
                activeOpacity={0.8}
              >
                <Text style={[styles.portionBtnText, { color: colors.text }]}>1x</Text>
              </TouchableOpacity>

              <TouchableOpacity
                style={[styles.portionBtn, { backgroundColor: colors.surfaceSecondary, borderColor: colors.border }]}
                onPress={() => {
                  setPortionMultiplier(1.5);
                  setPortionInputUnit('x');
                  setPortionInputValue('1.5');
                }}
                activeOpacity={0.8}
              >
                <Text style={[styles.portionBtnText, { color: colors.text }]}>1.5x</Text>
              </TouchableOpacity>

              <TouchableOpacity
                style={[styles.portionBtn, { backgroundColor: colors.primarySoft, borderColor: colors.primary }]}
                onPress={() => {
                  const next = Number((portionMultiplier + 0.25).toFixed(2));
                  setPortionMultiplier(next);
                  setPortionInputUnit('x');
                  setPortionInputValue(String(next));
                }}
                activeOpacity={0.8}
              >
                <Text style={[styles.portionBtnText, { color: colors.primary }]}>+ 0.25x</Text>
              </TouchableOpacity>
            </View>

            <View style={[styles.scaledNutritionCard, { backgroundColor: colors.primarySoft }]}>
              <Text style={[styles.scaledNutritionTitle, { color: colors.primary }]}>Scaled Nutrition</Text>
              <View style={styles.scaledNutritionGrid}>
                <View style={styles.scaledNutritionItem}>
                  <Text style={[styles.scaledNutritionValue, { color: colors.primary }]}>
                    {Math.round((selectedSavedMealForLog?.calories || 0) * effectivePortionMultiplier)}
                  </Text>
                  <Text style={[styles.scaledNutritionLabel, { color: colors.primary }]}>kcal</Text>
                </View>
                <View style={styles.scaledNutritionItem}>
                  <Text style={[styles.scaledNutritionValue, { color: colors.primary }]}>
                    {Math.round((selectedSavedMealForLog?.protein || 0) * effectivePortionMultiplier)}g
                  </Text>
                  <Text style={[styles.scaledNutritionLabel, { color: colors.primary }]}>P</Text>
                </View>
                <View style={styles.scaledNutritionItem}>
                  <Text style={[styles.scaledNutritionValue, { color: colors.primary }]}>
                    {Math.round((selectedSavedMealForLog?.carbs || 0) * effectivePortionMultiplier)}g
                  </Text>
                  <Text style={[styles.scaledNutritionLabel, { color: colors.primary }]}>C</Text>
                </View>
                <View style={styles.scaledNutritionItem}>
                  <Text style={[styles.scaledNutritionValue, { color: colors.primary }]}>
                    {Math.round((selectedSavedMealForLog?.fat || 0) * effectivePortionMultiplier)}g
                  </Text>
                  <Text style={[styles.scaledNutritionLabel, { color: colors.primary }]}>F</Text>
                </View>
              </View>
            </View>

            <View style={styles.sheetActions}>
              <Button
                title="Cancel"
                size="sm"
                variant="outline"
                onPress={() => {
                  setIsPortionAdjustOpen(false);
                  setSelectedSavedMealForLog(null);
                  setPortionMultiplier(1);
                  setPortionInputUnit('x');
                  setPortionInputValue('1');
                  setIsMealEntryModalOpen(true);
                }}
                style={styles.sheetBtn}
              />
              <Button title="Log Meal" size="sm" onPress={handleLogSavedMealWithPortion} style={styles.sheetBtn} />
            </View>
          </View>
        </View>
      </Modal>
    </KeyboardAvoidingView>
  );
}

function formatAmount(value) {
  if (!Number.isFinite(value)) return '0';
  return Math.abs(value) < 10 ? Number(value.toFixed(1)).toString() : Math.round(value).toString();
}

function joinPlan(values) {
  if (!Array.isArray(values) || values.length === 0) return DEFAULT_PROFILE.planLabel;
  return values.join(', ');
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  content: {
    paddingHorizontal: Spacing.md,
    gap: 10,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 4,
  },
  greeting: {
    ...Typography.body,
    marginBottom: 2,
  },
  greetingSub: {
    ...Typography.body,
    marginTop: 2,
  },
  userName: {
    fontSize: 22,
    fontWeight: '700',
  },
  headerActions: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.sm,
  },
  iconBtn: {
    width: 38,
    height: 38,
    borderRadius: 19,
    borderWidth: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  iconBtnText: {
    ...Typography.h3,
  },
  avatar: {
    width: 38,
    height: 38,
    borderRadius: 19,
    alignItems: 'center',
    justifyContent: 'center',
    overflow: 'hidden',
  },
  headerAvatarImage: {
    width: '100%',
    height: '100%',
    resizeMode: 'cover',
  },
  avatarText: {
    ...Typography.bodyMedium,
    fontWeight: '700',
  },
  calorieCard: {
    marginBottom: 8,
    borderWidth: 1,
    position: 'relative',
  },
  calorieOverlayShape: {
    position: 'absolute',
    right: -26,
    top: -24,
    width: 180,
    height: 150,
    borderRadius: 90,
    opacity: 0.5,
  },
  calorieHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 10,
  },
  liveBadge: {
    borderWidth: 1,
    borderRadius: BorderRadius.full,
    paddingHorizontal: 10,
    paddingVertical: 4,
  },
  liveBadgeText: {
    ...Typography.bodyMedium,
  },
  calorieRow: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: Spacing.md,
  },
  calNumber: {
    fontSize: 34,
    fontWeight: '700',
  },
  calLabel: {
    ...Typography.body,
    marginTop: 2,
  },
  calorieInfo: {
    flex: 1,
    gap: 6,
  },
  calorieTitle: {
    fontSize: 18,
    fontWeight: '700',
  },
  calorieSubtitle: {
    fontSize: 13,
    lineHeight: 18,
  },
  calorieStreakText: {
    ...Typography.caption,
    marginTop: 4,
  },
  remainingLabel: {
    ...Typography.h2,
  },
  remainingValue: {
    ...Typography.h2,
  },
  remainingSub: {
    ...Typography.caption,
  },
  badgeRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 6,
  },
  metaBadge: {
    borderRadius: BorderRadius.full,
    paddingHorizontal: 8,
    paddingVertical: 4,
  },
  metaBadgeText: {
    ...Typography.bodyMedium,
  },
  streakBadge: {
    alignSelf: 'flex-start',
    paddingHorizontal: Spacing.md,
    paddingVertical: Spacing.xs,
    borderRadius: BorderRadius.full,
    marginTop: Spacing.xs,
  },
  streakText: {
    ...Typography.h3,
  },
  editCalorieBtn: {
    borderRadius: BorderRadius.md,
    paddingHorizontal: 14,
    paddingVertical: 10,
    marginTop: 12,
    alignSelf: 'flex-start',
    height: 44,
    alignItems: 'center',
    justifyContent: 'center',
  },
  editCalorieTitle: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '600',
  },
  editCalorieSub: {
    fontSize: 11,
    color: 'rgba(255,255,255,0.86)',
    textTransform: 'uppercase',
    letterSpacing: 0.4,
  },
  xpCard: {
    borderWidth: 1,
    marginBottom: 8,
    ...Shadow.sm,
  },
  xpHeaderRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 6,
  },
  xpHeaderRight: {
    alignItems: 'flex-end',
    gap: 6,
  },
  xpHeaderMetaRight: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  xpDropdownArrow: {
    ...Typography.h2,
    lineHeight: 18,
  },
  xpLevelLabel: {
    ...Typography.captionMedium,
  },
  xpCollapsedSummaryRow: {
    flexDirection: 'row',
    justifyContent: 'flex-start',
    alignItems: 'center',
    marginBottom: 12,
  },
  xpCollapsedSummaryText: {
    ...Typography.captionMedium,
  },
  xpBadge: {
    borderRadius: BorderRadius.full,
    paddingHorizontal: 10,
    paddingVertical: 6,
  },
  xpBadgeText: {
    ...Typography.h3,
  },
  xpStatsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
    marginBottom: 12,
  },
  xpStatItem: {
    width: '48%',
    borderRadius: BorderRadius.md,
    padding: 8,
  },
  xpStatLabel: {
    ...Typography.caption,
    marginBottom: 3,
  },
  xpStatValue: {
    ...Typography.h2,
    fontSize: 16,
  },
  xpProgressWrap: {
    marginBottom: 10,
  },
  xpProgressLabel: {
    ...Typography.caption,
    marginBottom: 4,
  },
  xpBreakdownWrap: {
    gap: 5,
    marginBottom: 10,
  },
  xpBreakdownRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  xpBreakdownLabel: {
    ...Typography.caption,
  },
  xpBreakdownValue: {
    ...Typography.captionMedium,
  },
  xpReasonCard: {
    borderRadius: BorderRadius.md,
    padding: 8,
    marginBottom: 8,
  },
  xpMissedCard: {
    borderRadius: BorderRadius.md,
    padding: 8,
    marginBottom: 8,
  },
  xpReasonTitle: {
    ...Typography.bodyMedium,
    fontWeight: '700',
    marginBottom: 4,
  },
  xpReasonText: {
    ...Typography.caption,
    marginBottom: 2,
  },
  xpFooterText: {
    ...Typography.caption,
  },
  xpBadgeRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 6,
    marginTop: 8,
  },
  xpBadgeChip: {
    borderRadius: BorderRadius.full,
    paddingHorizontal: 8,
    paddingVertical: 5,
  },
  xpBadgeChipText: {
    ...Typography.captionMedium,
  },
  sectionHeaderRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginTop: 2,
  },
  sectionTitle: {
    fontSize: 17,
    fontWeight: '700',
  },
  sectionSub: {
    ...Typography.body,
    marginTop: 2,
  },
  editMacrosPill: {
    borderWidth: 1,
    borderRadius: BorderRadius.full,
    paddingHorizontal: 12,
    paddingVertical: 6,
  },
  editMacrosPillText: {
    ...Typography.h3,
  },
  macroRow: {
    flexDirection: 'row',
    gap: 8,
    marginBottom: 12,
  },
  macroCard: {
    flex: 1,
    padding: 10,
    minHeight: 96,
  },
  macroTopRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  macroLabel: {
    fontSize: 11,
    fontWeight: '600',
    marginBottom: 4,
  },
  macroPercent: {
    fontSize: 12,
    fontWeight: '600',
  },
  macroValue: {
    fontSize: 15,
    fontWeight: '700',
    marginBottom: 4,
  },
  macroBar: {
    marginTop: Spacing.xs,
  },
  waterCard: {
    padding: Spacing.md,
    marginBottom: 8,
  },
  waterHeader: {
    marginBottom: Spacing.md,
  },
  hydrationProgressSection: {
    marginBottom: Spacing.md,
  },
  hydrationStats: {
    flexDirection: 'row',
    alignItems: 'baseline',
    marginBottom: Spacing.xs,
  },
  hydrationValue: {
    fontSize: 22,
    fontWeight: '700',
  },
  hydrationTarget: {
    fontSize: 14,
    marginLeft: 6,
  },
  hydrationProgressBar: {
    display: 'none',
  },
  hydrationProgressTrack: {
    marginTop: 2,
    height: 9,
    borderRadius: BorderRadius.full,
    overflow: 'hidden',
  },
  hydrationProgressFill: {
    height: '100%',
    borderRadius: BorderRadius.full,
  },
  hydrationNutrientsCard: {
    borderWidth: 1,
    borderRadius: BorderRadius.md,
    padding: Spacing.sm,
    marginTop: Spacing.sm,
  },
  hydrationNutrientsTitle: {
    ...Typography.bodyMedium,
    marginBottom: Spacing.xs,
    fontWeight: '600',
  },
  hydrationNutrientsRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: Spacing.xs,
  },
  hydrationNutrientChip: {
    borderRadius: BorderRadius.full,
    paddingHorizontal: 10,
    paddingVertical: 6,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  hydrationNutrientLabel: {
    ...Typography.bodyMedium,
    fontWeight: '700',
  },
  hydrationNutrientValue: {
    ...Typography.bodyMedium,
    fontWeight: '600',
  },
  drinkTypeSection: {
    marginBottom: Spacing.md,
  },
  drinkTypeLabel: {
    ...Typography.bodyMedium,
    marginBottom: Spacing.xs,
  },
  drinkTypeGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 6,
  },
  drinkTypeBtn: {
    borderWidth: 1,
    borderRadius: BorderRadius.full,
    paddingHorizontal: 10,
    paddingVertical: 6,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    minHeight: 30,
  },
  drinkTypeEmoji: {
    fontSize: 14,
  },
  drinkTypeName: {
    ...Typography.captionMedium,
    fontWeight: '600',
  },
  waterButtonGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: Spacing.sm,
    marginBottom: Spacing.md,
  },
  waterBtn: {
    flex: 0.32,
    paddingHorizontal: Spacing.sm,
    paddingVertical: 10,
    borderRadius: BorderRadius.md,
    borderWidth: 1.5,
    alignItems: 'center',
    justifyContent: 'center',
  },
  waterBtnLabel: {
    fontSize: 14,
    fontWeight: '600',
  },
  customWaterBtn: {
    paddingHorizontal: Spacing.md,
    paddingVertical: 12,
    borderRadius: BorderRadius.md,
    alignItems: 'center',
    justifyContent: 'center',
    height: 44,
  },
  customWaterBtnText: {
    color: '#ffffff',
    fontSize: 14,
    fontWeight: '600',
  },
  waterInput: {
    borderWidth: 1,
    borderRadius: BorderRadius.md,
    paddingHorizontal: Spacing.md,
    paddingVertical: Spacing.sm,
    fontSize: 16,
    marginBottom: Spacing.md,
    height: 44,
  },
  hydrationInputRow: {
    flexDirection: 'row',
    alignItems: 'stretch',
    gap: Spacing.sm,
    marginBottom: Spacing.md,
  },
  hydrationAmountInput: {
    flex: 1,
    borderWidth: 1,
    borderRadius: BorderRadius.md,
    paddingHorizontal: Spacing.md,
    paddingVertical: Spacing.sm,
    fontSize: 16,
    height: 44,
  },
  hydrationUnitPicker: {
    borderWidth: 1,
    borderRadius: BorderRadius.md,
    paddingHorizontal: 6,
    justifyContent: 'center',
    minWidth: 160,
  },
  unitOption: {
    borderRadius: BorderRadius.full,
    paddingHorizontal: 10,
    paddingVertical: 8,
    marginHorizontal: 2,
    minWidth: 44,
    alignItems: 'center',
    justifyContent: 'center',
  },
  unitOptionText: {
    ...Typography.bodyMedium,
  },
  drinkTypeModalSection: {
    marginBottom: Spacing.md,
  },
  drinkTypeModalLabel: {
    ...Typography.body,
    fontWeight: '600',
    marginBottom: Spacing.xs,
  },
  drinkTypeModalGrid: {
    maxHeight: 220,
  },
  drinkTypeModalOption: {
    borderWidth: 1,
    borderRadius: BorderRadius.md,
    paddingHorizontal: Spacing.sm,
    paddingVertical: Spacing.sm,
    marginBottom: Spacing.xs,
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.sm,
  },
  drinkTypeModalEmoji: {
    fontSize: 18,
  },
  drinkTypeModalName: {
    ...Typography.body,
    fontWeight: '600',
  },
  drinkTypeModalNutrients: {
    ...Typography.bodyMedium,
    marginTop: 2,
  },
  waterUnitHint: {
    paddingHorizontal: Spacing.md,
    paddingVertical: Spacing.sm,
    marginBottom: Spacing.md,
    borderRadius: BorderRadius.md,
  },
  waterUnitText: {
    fontSize: 13,
    lineHeight: 18,
  },
  scanCta: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 14,
    paddingVertical: 14,
    borderRadius: BorderRadius.xl,
    marginBottom: 10,
    gap: 10,
    minHeight: 60,
    ...Shadow.sm,
    elevation: 6,
  },
  scanIconWrap: {
    width: 36,
    height: 36,
    borderRadius: 18,
    alignItems: 'center',
    justifyContent: 'center',
  },
  scanIcon: {
    fontSize: 18,
  },
  scanCtaText: {
    flex: 1,
  },
  scanTitle: {
    fontSize: 15,
    fontWeight: '700',
    color: '#ffffff',
  },
  scanSubtitle: {
    fontSize: 12,
    color: 'rgba(255,255,255,0.8)',
    marginTop: 1,
    lineHeight: 16,
  },
  scanArrow: {
    fontSize: 18,
    color: '#ffffff',
    fontWeight: '700',
  },
  favoritesCard: {
    marginBottom: 8,
  },
  favoritesCta: {
    borderWidth: 1,
    borderRadius: BorderRadius.full,
    paddingHorizontal: 12,
    paddingVertical: 7,
  },
  favoritesCtaText: {
    ...Typography.captionMedium,
    fontWeight: '700',
  },
  emptyFavoritesText: {
    ...Typography.body,
    marginTop: 2,
  },
  favoriteList: {
    marginTop: Spacing.sm,
    gap: Spacing.xs,
  },
  favoriteItem: {
    borderWidth: 1,
    borderRadius: BorderRadius.md,
    paddingHorizontal: Spacing.sm,
    paddingVertical: 8,
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.sm,
  },
  favoriteThumb: {
    width: 34,
    height: 34,
    borderRadius: 17,
    alignItems: 'center',
    justifyContent: 'center',
    overflow: 'hidden',
  },
  favoriteImage: {
    width: 34,
    height: 34,
    borderRadius: 17,
  },
  favoriteEmoji: {
    fontSize: 16,
  },
  favoriteInfo: {
    flex: 1,
  },
  favoriteName: {
    ...Typography.bodyMedium,
    fontWeight: '700',
  },
  favoriteMeta: {
    ...Typography.caption,
  },
  favoriteLogBtn: {
    borderRadius: BorderRadius.full,
    paddingHorizontal: 12,
    paddingVertical: 6,
  },
  favoriteLogBtnText: {
    ...Typography.captionMedium,
    color: '#ffffff',
  },
  emptyMealCard: {
    alignItems: 'center',
    paddingVertical: 16,
    marginTop: 4,
  },
  emptyMealIconWrap: {
    width: 58,
    height: 58,
    borderRadius: 29,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 8,
  },
  emptyMealIcon: {
    fontSize: 24,
  },
  emptyMealTitle: {
    ...Typography.h1,
    fontSize: 18,
    marginBottom: 6,
  },
  emptyMealSub: {
    ...Typography.body,
    textAlign: 'center',
    marginBottom: 10,
    lineHeight: 16,
    maxWidth: 430,
  },
  emptyMealBtn: {
    borderRadius: BorderRadius.full,
    paddingHorizontal: 20,
    paddingVertical: 9,
  },
  emptyMealBtnText: {
    ...Typography.h2,
    color: '#fff',
  },
  mealList: {
    gap: Spacing.md,
  },
  mealCard: {
    borderWidth: 1,
    borderRadius: BorderRadius.lg,
    paddingHorizontal: Spacing.sm,
    paddingVertical: 11,
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.sm,
    ...Shadow.sm,
  },
  mealThumb: {
    width: 42,
    height: 42,
    borderRadius: 21,
    alignItems: 'center',
    justifyContent: 'center',
    overflow: 'hidden',
    flexShrink: 0,
  },
  mealImage: {
    width: 42,
    height: 42,
    borderRadius: 21,
  },
  mealEmoji: {
    fontSize: 20,
  },
  mealInfo: {
    flex: 1,
  },
  mealName: {
    ...Typography.h3,
    marginBottom: 2,
  },
  mealMeta: {
    ...Typography.caption,
  },
  mealCalories: {
    ...Typography.body,
    fontSize: 13,
    fontWeight: '600',
  },
  insightsCard: {
    marginTop: 2,
  },
  insightHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    gap: Spacing.md,
  },
  insightHeaderText: {
    flex: 1,
  },
  insightTitle: {
    ...Typography.hero,
    fontSize: 21,
  },
  insightSub: {
    ...Typography.body,
    marginTop: 2,
  },
  insightHint: {
    ...Typography.body,
    marginTop: 4,
    lineHeight: 18,
  },
  insightSummaryRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 6,
    marginTop: 8,
  },
  summaryChip: {
    borderRadius: BorderRadius.full,
    paddingHorizontal: 8,
    paddingVertical: 4,
  },
  summaryChipText: {
    ...Typography.caption,
  },
  chevron: {
    ...Typography.h2,
    marginTop: 2,
  },
  insightSections: {
    marginTop: 10,
    gap: 12,
  },
  insightSectionBlock: {
    gap: 8,
  },
  insightSectionHead: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  insightSectionTitle: {
    ...Typography.h2,
  },
  insightSectionSub: {
    ...Typography.caption,
    marginTop: 2,
  },
  viewAllText: {
    ...Typography.captionMedium,
    textTransform: 'lowercase',
  },
  nutrientCard: {
    borderRadius: BorderRadius.lg,
    padding: 10,
  },
  nutrientHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 6,
  },
  nutrientMetaRight: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  modePill: {
    borderRadius: BorderRadius.full,
    paddingHorizontal: 8,
    paddingVertical: 4,
  },
  nutrientTitle: {
    ...Typography.h3,
    fontSize: 18,
    fontWeight: '700',
  },
  nutrientAuto: {
    ...Typography.bodyMedium,
    fontSize: 12,
  },
  infoIconBtn: {
    width: 22,
    height: 22,
    borderRadius: 11,
    borderWidth: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  infoIconText: {
    ...Typography.captionMedium,
    fontSize: 12,
  },
  nutrientStatusRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 4,
  },
  nutrientStatusText: {
    ...Typography.captionMedium,
  },
  nutrientPercentText: {
    ...Typography.caption,
  },
  nutrientNumbers: {
    ...Typography.caption,
    fontSize: 13,
    marginBottom: 6,
  },
  nutrientProgressBar: {
    marginBottom: 6,
  },
  nutrientBottomRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  nutrientActionRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  nutrientTarget: {
    ...Typography.caption,
    fontSize: 13,
  },
  nutrientReset: {
    ...Typography.caption,
    fontSize: 12,
  },
  nutrientEdit: {
    ...Typography.captionMedium,
    fontSize: 13,
  },
  goalCard: {
    marginTop: 4,
  },
  goalTitle: {
    ...Typography.hero,
    fontSize: 20,
    marginBottom: 4,
  },
  goalSubline: {
    ...Typography.h2,
    marginBottom: 8,
  },
  goalTagRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 6,
    marginBottom: 10,
  },
  goalTag: {
    borderRadius: BorderRadius.full,
    paddingHorizontal: 10,
    paddingVertical: 6,
  },
  goalTagText: {
    ...Typography.captionMedium,
  },
  goalMetricsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
    marginBottom: 8,
  },
  goalMetricCard: {
    width: '48%',
    borderRadius: BorderRadius.lg,
    padding: 10,
  },
  metricLabel: {
    ...Typography.caption,
    marginBottom: 4,
  },
  metricValue: {
    ...Typography.h2,
    fontSize: 16,
  },
  goalFootnote: {
    ...Typography.body,
    lineHeight: 18,
    marginBottom: 10,
  },
  goalEditBtn: {
    borderWidth: 1,
    borderRadius: BorderRadius.full,
    paddingHorizontal: 12,
    paddingVertical: 8,
    alignSelf: 'flex-start',
  },
  goalEditText: {
    ...Typography.h3,
  },
  overlay: {
    flex: 1,
    backgroundColor: 'rgba(15,23,42,0.42)',
    justifyContent: 'flex-end',
    padding: 10,
  },
  sheet: {
    borderRadius: BorderRadius.xl,
    padding: 12,
    maxHeight: '86%',
    ...Shadow.lg,
  },
  largeSheet: {
    borderRadius: BorderRadius.xl,
    padding: 12,
    height: '88%',
    ...Shadow.lg,
  },
  smallSheet: {
    borderRadius: BorderRadius.xl,
    padding: 12,
    ...Shadow.lg,
  },
  sheetTitle: {
    ...Typography.h2,
    marginBottom: 4,
  },
  sheetSub: {
    ...Typography.body,
    marginBottom: 8,
  },
  goalTypeRow: {
    flexDirection: 'row',
    gap: 6,
    marginBottom: 8,
  },
  goalTypeChip: {
    flex: 1,
    borderWidth: 1,
    borderRadius: BorderRadius.full,
    minHeight: 36,
    alignItems: 'center',
    justifyContent: 'center',
  },
  goalTypeText: {
    ...Typography.captionMedium,
  },
  fieldLabel: {
    ...Typography.captionMedium,
    marginBottom: 4,
  },
  stepperRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 8,
    marginBottom: 10,
  },
  stepperBtn: {
    borderWidth: 1,
    borderRadius: BorderRadius.full,
    minWidth: 72,
    minHeight: 36,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 10,
  },
  stepperText: {
    ...Typography.bodyMedium,
  },
  stepperValue: {
    ...Typography.h2,
    flex: 1,
    textAlign: 'center',
  },
  goalErrorText: {
    ...Typography.caption,
    marginTop: 8,
  },
  sheetActions: {
    flexDirection: 'row',
    gap: 8,
    marginTop: 8,
  },
  stickyActions: {
    paddingTop: 8,
    borderTopWidth: 1,
    borderTopColor: 'rgba(148,163,184,0.25)',
  },
  sheetBtn: {
    flex: 1,
  },
  largeSheetBody: {
    paddingTop: 4,
    paddingBottom: 12,
    gap: 8,
  },
  autoMacroBtn: {
    borderWidth: 1,
    borderRadius: BorderRadius.full,
    minHeight: 38,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 8,
  },
  autoMacroText: {
    ...Typography.h3,
  },
  macroEditorRow: {
    marginBottom: 8,
  },
  inlineStepper: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  inlineStepBtn: {
    borderWidth: 1,
    borderRadius: BorderRadius.full,
    minWidth: 60,
    minHeight: 38,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 8,
  },
  inlineStepText: {
    ...Typography.bodyMedium,
  },
  inlineValueBox: {
    flex: 1,
    minHeight: 38,
    borderWidth: 1,
    borderRadius: BorderRadius.md,
    justifyContent: 'center',
    paddingHorizontal: 10,
  },
  inlineValueText: {
    ...Typography.h3,
  },
  scanSheet: {
    borderRadius: BorderRadius.xl,
    padding: 12,
    ...Shadow.lg,
  },
  sampleMealList: {
    gap: 6,
    marginBottom: 8,
  },
  scanTabsRow: {
    flexDirection: 'row',
    gap: 6,
    marginBottom: 10,
  },
  scanTabBtn: {
    flex: 1,
    borderWidth: 1,
    borderRadius: BorderRadius.full,
    minHeight: 36,
    alignItems: 'center',
    justifyContent: 'center',
  },
  scanTabText: {
    ...Typography.bodyMedium,
    fontWeight: '700',
  },
  sampleMealItem: {
    borderWidth: 1,
    borderRadius: BorderRadius.md,
    minHeight: 48,
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 8,
  },
  sampleMealMain: {
    flex: 1,
    justifyContent: 'center',
  },
  sampleMealSaveBtn: {
    borderWidth: 1,
    borderRadius: BorderRadius.full,
    paddingHorizontal: 10,
    paddingVertical: 6,
  },
  sampleMealSaveText: {
    ...Typography.captionMedium,
  },
  sampleMealName: {
    ...Typography.bodyMedium,
    marginBottom: 1,
  },
  sampleMealMeta: {
    ...Typography.caption,
  },
  fullWidth: {
    width: '100%',
  },
  mealDetailRow: {
    minHeight: 36,
    borderBottomWidth: 1,
    borderBottomColor: '#e2e8f0',
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  detailLabel: {
    ...Typography.body,
  },
  detailValue: {
    ...Typography.bodyMedium,
  },
  favoriteToggleBtn: {
    marginTop: 12,
    borderWidth: 1,
    borderRadius: BorderRadius.md,
    minHeight: 42,
    alignItems: 'center',
    justifyContent: 'center',
  },
  favoriteToggleBtnText: {
    ...Typography.bodyMedium,
    fontWeight: '700',
  },
  fullModalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 10,
    marginTop: 2,
  },
  closeCircle: {
    width: 40,
    height: 40,
    borderRadius: 20,
    borderWidth: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  closeCircleText: {
    ...Typography.h1,
    fontSize: 22,
  },
  targetInput: {
    borderWidth: 1,
    borderRadius: BorderRadius.md,
    minHeight: 38,
    paddingHorizontal: 10,
    marginTop: 6,
    marginBottom: 4,
    ...Typography.body,
  },
  editTargetRow: {
    borderRadius: BorderRadius.lg,
    padding: 10,
    marginBottom: 8,
  },
  editTargetHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 6,
  },
  editTargetTitle: {
    ...Typography.bodyMedium,
    fontWeight: '700',
  },
  editTargetMode: {
    ...Typography.captionMedium,
  },
  infoBodyText: {
    ...Typography.body,
    lineHeight: 20,
  },
  mealEntryOptions: {
    gap: Spacing.md,
    paddingVertical: Spacing.md,
  },
  mealEntryOption: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.md,
    padding: Spacing.md,
    borderRadius: BorderRadius.lg,
    borderWidth: 1,
  },
  mealEntryOptionIcon: {
    fontSize: 32,
  },
  mealEntryOptionText: {
    flex: 1,
  },
  mealEntryOptionTitle: {
    ...Typography.h3,
    marginBottom: 2,
  },
  mealEntryOptionSub: {
    ...Typography.caption,
  },
  savedMealsList: {
    paddingVertical: Spacing.md,
    gap: Spacing.sm,
  },
  savedMealItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.md,
    padding: Spacing.md,
    borderRadius: BorderRadius.lg,
  },
  savedMealImage: {
    width: 60,
    height: 60,
    borderRadius: BorderRadius.md,
  },
  savedMealInfo: {
    flex: 1,
  },
  savedMealName: {
    ...Typography.bodyMedium,
    fontWeight: '700',
    marginBottom: 4,
  },
  savedMealMeta: {
    ...Typography.caption,
    marginBottom: 2,
  },
  savedMealActions: {
    flexDirection: 'row',
    gap: 6,
  },
  savedMealActionBtn: {
    minWidth: 40,
    height: 36,
    borderRadius: BorderRadius.md,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 8,
  },
  savedMealActionBtnText: {
    color: '#fff',
    fontWeight: '600',
    fontSize: 12,
  },
  portionCard: {
    borderRadius: BorderRadius.lg,
    padding: Spacing.md,
    alignItems: 'center',
    marginVertical: Spacing.md,
  },
  portionLabel: {
    ...Typography.caption,
    marginBottom: 4,
  },
  portionValue: {
    fontSize: 32,
    fontWeight: '700',
  },
  portionUnitTabs: {
    width: '100%',
    flexDirection: 'row',
    gap: 6,
    marginTop: 8,
  },
  portionUnitBtn: {
    flex: 1,
    minHeight: 34,
    borderWidth: 1,
    borderRadius: BorderRadius.full,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 8,
  },
  portionUnitBtnText: {
    ...Typography.captionMedium,
  },
  portionInput: {
    width: '100%',
    borderWidth: 1,
    borderRadius: BorderRadius.md,
    minHeight: 40,
    marginTop: 8,
    paddingHorizontal: 10,
    ...Typography.body,
  },
  portionControlRow: {
    flexDirection: 'row',
    gap: 6,
    marginBottom: Spacing.md,
  },
  portionBtn: {
    flex: 1,
    paddingHorizontal: 8,
    paddingVertical: 10,
    borderRadius: BorderRadius.md,
    borderWidth: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  portionBtnText: {
    ...Typography.bodyMedium,
    fontWeight: '600',
    fontSize: 12,
  },
  scaledNutritionCard: {
    borderRadius: BorderRadius.lg,
    padding: Spacing.md,
    marginBottom: Spacing.md,
  },
  scaledNutritionTitle: {
    ...Typography.h3,
    marginBottom: Spacing.sm,
  },
  scaledNutritionGrid: {
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
  scaledNutritionItem: {
    alignItems: 'center',
  },
  scaledNutritionValue: {
    ...Typography.h2,
    marginBottom: 2,
  },
  scaledNutritionLabel: {
    ...Typography.caption,
  },
});

