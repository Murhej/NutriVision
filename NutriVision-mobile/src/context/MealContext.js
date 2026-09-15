import React, { createContext, useContext, useState, useCallback, useMemo } from 'react';
import { apiRequest } from '../api/client';
import { useAuth } from './AuthContext';

const MealContext = createContext(null);

const NUTRIENT_ALIASES = {
  calories: ['calories', 'Calories', 'kcal'],
  protein: ['protein', 'protein_g', 'Protein'],
  carbs: ['carbs', 'carbs_g', 'Carbs', 'carbohydrates', 'carbohydrate_g'],
  fat: ['fat', 'fat_g', 'Fat', 'total_fat_g'],
  fiber: ['fiber', 'fiber_g', 'dietary_fiber', 'dietary_fiber_g'],
  sugar: ['sugar', 'sugar_g', 'sugars', 'total_sugars_g'],
  water: ['water', 'water_g', 'water_ml', 'Water'],
  choline: ['choline', 'choline_mg', 'Choline'],
  vitaminA: ['vitaminA', 'VitaminA', 'Vitamin A, RAE'],
  vitaminB1: ['vitaminB1', 'thiamin', 'Thiamin'],
  vitaminB2: ['vitaminB2', 'riboflavin', 'Riboflavin'],
  vitaminB3: ['vitaminB3', 'niacin', 'Niacin'],
  vitaminB5: ['vitaminB5', 'pantothenic_acid', 'Pantothenic acid'],
  vitaminB6: ['vitaminB6', 'vitamin_b6', 'Vitamin B-6'],
  vitaminB7: ['vitaminB7', 'biotin', 'Biotin'],
  vitaminB9: ['vitaminB9', 'folate', 'Folate, total'],
  vitaminB12: ['vitaminB12', 'vitamin_b12', 'Vitamin B-12'],
  vitaminC: ['vitaminC', 'VitaminC', 'Vitamin C, total ascorbic acid'],
  vitaminD: ['vitaminD', 'VitaminD', 'Vitamin D (D2 + D3)'],
  vitaminE: ['vitaminE', 'VitaminE', 'Vitamin E (alpha-tocopherol)'],
  vitaminK: ['vitaminK', 'VitaminK', 'Vitamin K (phylloquinone)'],
  calcium: ['calcium', 'Calcium', 'Calcium, Ca'],
  iron: ['iron', 'Iron', 'Iron, Fe'],
  magnesium: ['magnesium', 'Magnesium', 'Magnesium, Mg'],
  potassium: ['potassium', 'Potassium', 'Potassium, K'],
  sodium: ['sodium', 'sodium_mg', 'Sodium', 'Sodium, Na'],
  zinc: ['zinc', 'Zinc', 'Zinc, Zn'],
  phosphorus: ['phosphorus', 'Phosphorus', 'Phosphorus, P'],
  chloride: ['chloride', 'chloride_mg', 'Chloride', 'Chloride, Cl'],
  sulfur: ['sulfur', 'sulfur_mg', 'Sulfur'],
  copper: ['copper', 'Copper', 'Copper, Cu'],
  manganese: ['manganese', 'Manganese', 'Manganese, Mn'],
  selenium: ['selenium', 'Selenium', 'Selenium, Se'],
  iodine: ['iodine', 'iodine_mcg', 'Iodine', 'Iodine, I'],
  fluoride: ['fluoride', 'fluoride_mg', 'Fluoride'],
  chromium: ['chromium', 'chromium_mcg', 'Chromium', 'Chromium, Cr'],
  molybdenum: ['molybdenum', 'molybdenum_mcg', 'Molybdenum', 'Molybdenum, Mo'],
  leucine: ['leucine', 'Leucine'],
  isoleucine: ['isoleucine', 'Isoleucine'],
  valine: ['valine', 'Valine'],
  lysine: ['lysine', 'Lysine'],
  methionine: ['methionine', 'Methionine'],
  cysteine: ['cysteine', 'Cysteine'],
  methionineCysteine: ['methionineCysteine', 'methionine_cysteine', 'Methionine + Cysteine'],
  phenylalanine: ['phenylalanine', 'Phenylalanine'],
  tyrosine: ['tyrosine', 'Tyrosine'],
  phenylalanineTyrosine: ['phenylalanineTyrosine', 'phenylalanine_tyrosine', 'Phenylalanine + Tyrosine'],
  threonine: ['threonine', 'Threonine'],
  tryptophan: ['tryptophan', 'Tryptophan'],
  histidine: ['histidine', 'Histidine'],
  omega3: ['omega3', 'omega_3', 'omega3_g', 'Omega3ALA'],
  omega6: ['omega6', 'omega_6', 'omega6_g', 'Omega6LA'],
  saturatedFat: ['saturatedFat', 'saturated_fat', 'saturated_fat_g'],
  monounsaturatedFat: ['monounsaturatedFat', 'monounsaturated_fat', 'monounsaturated_fat_g'],
  polyunsaturatedFat: ['polyunsaturatedFat', 'polyunsaturated_fat', 'polyunsaturated_fat_g'],
  transFat: ['transFat', 'trans_fat', 'trans_fat_g'],
};

const VITAMIN_KEYS = ['vitaminA', 'vitaminB1', 'vitaminB2', 'vitaminB3', 'vitaminB5', 'vitaminB6', 'vitaminB7', 'vitaminB9', 'vitaminB12', 'vitaminC', 'vitaminD', 'vitaminE', 'vitaminK'];
const MINERAL_KEYS = [
  'calcium',
  'phosphorus',
  'magnesium',
  'sodium',
  'potassium',
  'chloride',
  'sulfur',
  'iron',
  'zinc',
  'iodine',
  'selenium',
  'copper',
  'manganese',
  'fluoride',
  'chromium',
  'molybdenum',
];
const AMINO_ACID_KEYS = [
  'histidine',
  'isoleucine',
  'leucine',
  'lysine',
  'methionineCysteine',
  'phenylalanineTyrosine',
  'threonine',
  'tryptophan',
  'valine',
  'methionine',
  'cysteine',
  'phenylalanine',
  'tyrosine',
];
const FATTY_ACID_KEYS = ['omega3', 'omega6', 'saturatedFat', 'monounsaturatedFat', 'polyunsaturatedFat', 'transFat'];
const CORE_NUTRIENT_KEYS = ['calories', 'protein', 'carbs', 'fat', 'fiber', 'sugar', 'water', 'choline'];

// Drink type definitions with optional nutrient content for minerals/electrolytes
const DRINK_TYPES = {
  plain: {
    name: 'Plain Water',
    emoji: '💧',
    nutrients: {},
  },
  lemon: {
    name: 'Lemon Water',
    emoji: '🍋',
    nutrients: { vitaminC: 5 }, // per 250ml
  },
  mineral: {
    name: 'Mineral Water',
    emoji: '⛰️',
    nutrients: { calcium: 50, magnesium: 10, sodium: 30 }, // per 250ml
  },
  electrolyte: {
    name: 'Electrolyte Water',
    emoji: '⚡',
    nutrients: { sodium: 200, potassium: 150, magnesium: 40, chloride: 300 }, // per 250ml
  },
  sparkling: {
    name: 'Sparkling Water',
    emoji: '✨',
    nutrients: {},
  },
  tea: {
    name: 'Tea',
    emoji: '🍵',
    nutrients: { vitaminC: 3 }, // per 250ml
    caffeine: 25, // mg per 250ml
  },
  coffee: {
    name: 'Coffee',
    emoji: '☕',
    nutrients: {},
    caffeine: 95, // mg per 250ml
  },
};

// Unit conversion to ml (base unit)
const UNIT_TO_ML = {
  ml: 1,
  l: 1000,
  oz: 29.5735,
  cup: 236.588,
};

function convertToMl(amount, unit) {
  const multiplier = UNIT_TO_ML[unit?.toLowerCase()] || 1;
  return Number((amount * multiplier).toFixed(2));
}

function getHydrationTarget(metrics) {
  // Base: 30 ml per kg of body weight
  let baseTarget = metrics.weightKg * 30;
  // Activity adjustment: +500ml per activity level above 1.0
  const activityBonus = Math.max(0, (metrics.activityMultiplier - 1) * 500);
  return Math.round(baseTarget + activityBonus);
}

const DEFAULT_PROFILE_METRICS = {
  age: 25,
  gender: 'female',
  heightCm: 168,
  weightKg: 68,
  activityMultiplier: 1.35,
  goalType: 'maintain',
};

const GOAL_CALORIE_ADJUSTMENT = {
  lose: -400,
  maintain: 0,
  gain: 300,
  muscle: 250,
};

const GOAL_MACRO_SPLIT = {
  lose: { protein: 0.35, carbs: 0.3, fat: 0.35 },
  maintain: { protein: 0.28, carbs: 0.42, fat: 0.3 },
  gain: { protein: 0.25, carbs: 0.5, fat: 0.25 },
  muscle: { protein: 0.33, carbs: 0.42, fat: 0.25 },
};

const XP_BUCKETS = {
  calories: 100,
  macros: 100,
  fiber: 25,
  choline: 20,
  water: 25,
  vitamins: 100,
  majorMinerals: 70,
  traceMinerals: 70,
  aminoAcids: 80,
  fattyAcids: 30,
  bonus: 30,
};

const GOAL_MACRO_XP_WEIGHTS = {
  maintain: { protein: 33, carbs: 33, fat: 34 },
  lose: { protein: 45, carbs: 25, fat: 30 },
  muscle: { protein: 50, carbs: 30, fat: 20 },
  gain: { protein: 40, carbs: 35, fat: 25 },
};

const VITAMIN_XP_WEIGHTS = {
  vitaminA: 8,
  vitaminC: 8,
  vitaminD: 12,
  vitaminE: 6,
  vitaminK: 6,
  vitaminB1: 6,
  vitaminB2: 6,
  vitaminB3: 6,
  vitaminB5: 6,
  vitaminB6: 8,
  vitaminB7: 5,
  vitaminB9: 10,
  vitaminB12: 13,
};

const MAJOR_MINERAL_KEYS = ['calcium', 'phosphorus', 'magnesium', 'sodium', 'potassium', 'chloride', 'sulfur'];
const TRACE_MINERAL_KEYS = ['iron', 'zinc', 'iodine', 'selenium', 'copper', 'manganese', 'fluoride', 'chromium', 'molybdenum'];

const MAJOR_MINERAL_XP_WEIGHTS = {
  calcium: 12,
  phosphorus: 8,
  magnesium: 12,
  sodium: 8,
  potassium: 12,
  chloride: 8,
  sulfur: 10,
};

const TRACE_MINERAL_XP_WEIGHTS = {
  iron: 12,
  zinc: 10,
  iodine: 8,
  selenium: 8,
  copper: 8,
  manganese: 6,
  fluoride: 4,
  chromium: 6,
  molybdenum: 8,
};

const ESSENTIAL_AMINO_KEYS = ['histidine', 'isoleucine', 'leucine', 'lysine', 'methionineCysteine', 'phenylalanineTyrosine', 'threonine', 'tryptophan', 'valine'];
const AMINO_XP_WEIGHTS = {
  histidine: 8,
  isoleucine: 9,
  leucine: 11,
  lysine: 9,
  methionineCysteine: 9,
  phenylalanineTyrosine: 9,
  threonine: 8,
  tryptophan: 8,
  valine: 9,
};

const FATTY_XP_WEIGHTS = {
  omega3: 18,
  omega6: 12,
};

const XP_LEVEL_STEP = 500;
const STRONG_DAY_XP = 100;

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function roundXp(value) {
  return Number(clamp(Number(value) || 0, 0, 100000).toFixed(2));
}

function ratioScoreBand(ratio) {
  if (!Number.isFinite(ratio) || ratio <= 0) return 0;
  if (ratio >= 0.95 && ratio <= 1.05) return 1;
  if ((ratio >= 0.9 && ratio < 0.95) || (ratio > 1.05 && ratio <= 1.1)) return 0.75;
  if ((ratio >= 0.8 && ratio < 0.9) || (ratio > 1.1 && ratio <= 1.2)) return 0.4;
  if (ratio < 0.8 || ratio > 1.2) return 0.15;
  return 0;
}

function ratioScoreBandMicronutrient(ratio) {
  if (!Number.isFinite(ratio) || ratio <= 0) return 0;
  if (ratio >= 1) return 1;
  if (ratio >= 0.8) return 0.75;
  if (ratio >= 0.5) return 0.4;
  return 0.1;
}

function scoreCalorieByGoal(consumed, target, goalType) {
  const safeTarget = toNumber(target);
  const safeConsumed = toNumber(consumed);
  if (safeTarget <= 0) {
    return { points: 0, reason: 'No calorie target available' };
  }

  const ratio = safeConsumed / safeTarget;
  const delta = safeConsumed - safeTarget;

  if (goalType === 'maintain') {
    if (ratio >= 0.97 && ratio <= 1.03) return { points: XP_BUCKETS.calories, reason: 'Great calorie alignment for maintenance' };
    if ((ratio >= 0.94 && ratio < 0.97) || (ratio > 1.03 && ratio <= 1.06)) return { points: XP_BUCKETS.calories * 0.75, reason: 'Calories close to maintenance range' };
    if ((ratio >= 0.9 && ratio < 0.94) || (ratio > 1.06 && ratio <= 1.1)) return { points: XP_BUCKETS.calories * 0.5, reason: 'Calories moderately off maintenance range' };
    return { points: XP_BUCKETS.calories * 0.15, reason: 'Calories far from maintenance target' };
  }

  if (goalType === 'lose') {
    if (ratio < 0.7) return { points: XP_BUCKETS.calories * 0.2, reason: 'Under-eating too aggressively reduced calorie XP' };
    if (delta <= 0 && ratio >= 0.85) return { points: XP_BUCKETS.calories, reason: 'Calories in fat-loss range' };
    if (delta > 0 && delta <= 50) return { points: XP_BUCKETS.calories * 0.5, reason: 'Slight calorie overshoot reduced fat-loss XP' };
    if (delta > 50 && delta <= 150) return { points: XP_BUCKETS.calories * 0.25, reason: 'Calorie overshoot reduced fat-loss XP' };
    if (delta > 150 && delta <= 250) return { points: XP_BUCKETS.calories * 0.1, reason: 'Large calorie overshoot for fat-loss goal' };
    if (delta > 250) return { points: 0, reason: 'Calories too high for fat-loss goal' };
    return { points: XP_BUCKETS.calories * 0.6, reason: 'Calories below fat-loss range' };
  }

  if (goalType === 'muscle' || goalType === 'gain') {
    if (ratio >= 0.95 && ratio <= 1.1) return { points: XP_BUCKETS.calories, reason: 'Calories support growth goal' };
    if (delta < 0 && Math.abs(delta) <= 100) return { points: XP_BUCKETS.calories * 0.6, reason: 'Calories slightly low for growth goal' };
    if (delta < 0 && Math.abs(delta) <= 250) return { points: XP_BUCKETS.calories * 0.3, reason: 'Calories too low for growth goal' };
    if (delta < -250) return { points: XP_BUCKETS.calories * 0.15, reason: 'Calories much too low for growth goal' };
    if (ratio > 1.1 && ratio <= 1.2) return { points: XP_BUCKETS.calories * 0.8, reason: 'Slight calorie surplus acceptable for growth' };
    return { points: XP_BUCKETS.calories * 0.5, reason: 'Calorie surplus larger than ideal range' };
  }

  return { points: XP_BUCKETS.calories * ratioScoreBand(ratio), reason: 'Calories scored with standard goal band' };
}

function scoreWeightedNutrients({ totals, targets, weights, keys, bandFn, goalType = 'maintain', groupType = 'micros' }) {
  const details = [];
  let points = 0;

  keys.forEach((key) => {
    const weight = toNumber(weights[key]);
    const consumed = toNumber(totals[key]);
    const target = toNumber(targets[key]);
    if (weight <= 0 || target <= 0) return;

    const ratio = consumed / target;
    let multiplier = bandFn(ratio);

    if (groupType === 'amino' && goalType === 'muscle') {
      if (['leucine', 'isoleucine', 'valine', 'lysine'].includes(key)) {
        multiplier *= 1.15;
      }
      if (key === 'leucine' && ratio < 0.85) {
        multiplier *= 0.2;
      }
    }

    const earned = clamp(weight * multiplier, 0, weight);
    points += earned;
    details.push({ key, weight, earned: roundXp(earned), ratio: Number(ratio.toFixed(3)) });
  });

  return {
    points: roundXp(points),
    details,
  };
}

function sortDateKeysAsc(dateKeys) {
  return [...dateKeys].sort((a, b) => new Date(a) - new Date(b));
}

function previousDateKey(dateKey) {
  const d = new Date(dateKey);
  d.setDate(d.getDate() - 1);
  return toDateKey(d);
}

function toNumber(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

function pickFromAliases(source, aliases) {
  for (const alias of aliases) {
    if (source && Object.prototype.hasOwnProperty.call(source, alias)) {
      return toNumber(source[alias]);
    }
  }
  return 0;
}

function inferGoalType(goalText = '') {
  const lower = String(goalText || '').toLowerCase();
  if (lower.includes('muscle') || lower.includes('lean') || lower.includes('bulk')) return 'muscle';
  if (lower.includes('lose') || lower.includes('fat')) return 'lose';
  if (lower.includes('gain') || lower.includes('build')) return 'gain';
  return 'maintain';
}

function calculateBmr({ age, gender, heightCm, weightKg }) {
  const base = 10 * weightKg + 6.25 * heightCm - 5 * age;
  return String(gender || '').toLowerCase() === 'male' ? base + 5 : base - 161;
}

function parseMetric(value, fallback) {
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function normalizeProfileMetrics(profile = {}) {
  const goalType = ['lose', 'maintain', 'gain', 'muscle'].includes(profile?.goalType)
    ? profile.goalType
    : inferGoalType(profile?.goal || '');
  return {
    age: parseMetric(profile?.age, DEFAULT_PROFILE_METRICS.age),
    gender: String(profile?.gender || DEFAULT_PROFILE_METRICS.gender).toLowerCase(),
    heightCm: parseMetric(profile?.heightCm ?? profile?.height, DEFAULT_PROFILE_METRICS.heightCm),
    weightKg: parseMetric(profile?.weightKg ?? profile?.weight, DEFAULT_PROFILE_METRICS.weightKg),
    activityMultiplier: Math.max(1.1, Math.min(2.4, parseMetric(profile?.activityMultiplier, DEFAULT_PROFILE_METRICS.activityMultiplier))),
    goalType,
  };
}

function buildTargetStatus(consumed, target) {
  const consumedValue = toNumber(consumed);
  const targetValue = toNumber(target);
  const remaining = Math.max(0, targetValue - consumedValue);
  const percent = targetValue > 0 ? Math.round((consumedValue / targetValue) * 100) : 0;
  const ratio = targetValue > 0 ? consumedValue / targetValue : 0;
  let status = 'low';
  if (ratio > 1.05) status = 'over';
  else if (ratio >= 0.75) status = 'on-track';
  return {
    consumed: Number(consumedValue.toFixed(2)),
    target: Number(targetValue.toFixed(2)),
    remaining: Number(remaining.toFixed(2)),
    percentReached: Math.max(0, percent),
    status,
  };
}

function pickGroupValue(input, nestedKey, aliases) {
  return pickFromAliases(input[nestedKey] || {}, aliases);
}

function normalizeMealNutrition(input = {}) {
  const vitaminsInput = input.vitamins || {};
  const mineralsInput = input.minerals || {};
  const aminoAcidsInput = input.aminoAcids || input.amino_acids || {};
  const fattyAcidsInput = input.fattyAcids || input.fatty_acids || {};
  const flat = {
    ...(input || {}),
    ...vitaminsInput,
    ...mineralsInput,
    ...aminoAcidsInput,
    ...fattyAcidsInput,
  };

  const vitamins = Object.fromEntries(
    VITAMIN_KEYS.map((key) => [key, pickFromAliases(flat, NUTRIENT_ALIASES[key]) || pickGroupValue(input, 'vitamins', NUTRIENT_ALIASES[key])])
  );

  const minerals = Object.fromEntries(
    MINERAL_KEYS.map((key) => [key, pickFromAliases(flat, NUTRIENT_ALIASES[key]) || pickGroupValue(input, 'minerals', NUTRIENT_ALIASES[key])])
  );

  const aminoAcids = Object.fromEntries(
    AMINO_ACID_KEYS.map((key) => [key, pickFromAliases(flat, NUTRIENT_ALIASES[key]) || pickGroupValue(input, 'aminoAcids', NUTRIENT_ALIASES[key]) || pickGroupValue(input, 'amino_acids', NUTRIENT_ALIASES[key])])
  );

  if (!toNumber(aminoAcids.methionineCysteine)) {
    aminoAcids.methionineCysteine = toNumber(aminoAcids.methionine) + toNumber(aminoAcids.cysteine);
  }
  if (!toNumber(aminoAcids.phenylalanineTyrosine)) {
    aminoAcids.phenylalanineTyrosine = toNumber(aminoAcids.phenylalanine) + toNumber(aminoAcids.tyrosine);
  }

  const fattyAcids = Object.fromEntries(
    FATTY_ACID_KEYS.map((key) => [key, pickFromAliases(flat, NUTRIENT_ALIASES[key]) || pickGroupValue(input, 'fattyAcids', NUTRIENT_ALIASES[key]) || pickGroupValue(input, 'fatty_acids', NUTRIENT_ALIASES[key])])
  );

  const core = {
    calories: Math.round(toNumber(pickFromAliases(flat, NUTRIENT_ALIASES.calories))),
    protein: Math.round(toNumber(pickFromAliases(flat, NUTRIENT_ALIASES.protein))),
    carbs: Math.round(toNumber(pickFromAliases(flat, NUTRIENT_ALIASES.carbs))),
    fat: Math.round(toNumber(pickFromAliases(flat, NUTRIENT_ALIASES.fat))),
    fiber: Number(toNumber(pickFromAliases(flat, NUTRIENT_ALIASES.fiber)).toFixed(2)),
    sugar: Number(toNumber(pickFromAliases(flat, NUTRIENT_ALIASES.sugar)).toFixed(2)),
    water: Number(toNumber(pickFromAliases(flat, NUTRIENT_ALIASES.water)).toFixed(2)),
    choline: Number(toNumber(pickFromAliases(flat, NUTRIENT_ALIASES.choline)).toFixed(2)),
  };

  const nutrients = {
    ...core,
    ...vitamins,
    ...minerals,
    ...aminoAcids,
    ...fattyAcids,
  };

  return {
    ...core,
    vitamins,
    minerals,
    aminoAcids,
    fattyAcids,
    nutrients,
    fullNutrition: {
      ...core,
      vitamins,
      minerals,
      aminoAcids,
      fattyAcids,
      nutrients,
    },
  };
}

function toDateKey(date) {
  return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}-${String(date.getDate()).padStart(2, '0')}`;
}

export function MealProvider({ children }) {
  const { profile, achievements } = useAuth();
  // meals is an array of meal objects; each has a dateKey derived from timestamp
  const [meals, setMeals] = useState([]);
  // savedMeals: store meal templates with full nutrition data for reuse
  const [savedMeals, setSavedMeals] = useState([]);
  // favoriteMealIds: set of saved meal IDs that are marked as favorites
  const [favoriteMealIds, setFavoriteMealIds] = useState(new Set());
  // hydrationLogs: track water and drink intake with optional nutrient payloads
  const [hydrationLogs, setHydrationLogs] = useState([]);
  // instant XP sources such as corrections or manual rewards
  const [instantXpEvents, setInstantXpEvents] = useState([]);
  const [correctionSubmissions, setCorrectionSubmissions] = useState(0);

  const personalizedTargets = useMemo(() => {
    const metrics = normalizeProfileMetrics(profile || {});
    const bmr = calculateBmr(metrics);
    const tdee = bmr * metrics.activityMultiplier;
    const calories = Math.max(1200, Math.round(tdee + (GOAL_CALORIE_ADJUSTMENT[metrics.goalType] || 0)));
    const macroSplit = GOAL_MACRO_SPLIT[metrics.goalType] || GOAL_MACRO_SPLIT.maintain;
    const macros = {
      protein: Math.round((calories * macroSplit.protein) / 4),
      carbs: Math.round((calories * macroSplit.carbs) / 4),
      fat: Math.round((calories * macroSplit.fat) / 9),
    };

    const vitamins = {
      vitaminA: metrics.gender === 'male' ? 900 : 700,
      vitaminB1: 1.2,
      vitaminB2: 1.3,
      vitaminB3: metrics.gender === 'male' ? 16 : 14,
      vitaminB5: 5,
      vitaminB6: 1.3,
      vitaminB7: 30,
      vitaminB9: 400,
      vitaminB12: 2.4,
      vitaminC: metrics.gender === 'male' ? 90 : 75,
      vitaminD: 15,
      vitaminE: 15,
      vitaminK: metrics.gender === 'male' ? 120 : 90,
    };

    const minerals = {
      calcium: 1000,
      phosphorus: 700,
      magnesium: metrics.gender === 'male' ? 420 : 320,
      sodium: 2300,
      potassium: metrics.gender === 'male' ? 3400 : 2600,
      chloride: 2300,
      sulfur: Math.round(metrics.weightKg * 13),
      iron: metrics.gender === 'male' ? 8 : 18,
      zinc: metrics.gender === 'male' ? 11 : 8,
      iodine: 150,
      selenium: 55,
      copper: 0.9,
      manganese: metrics.gender === 'male' ? 2.3 : 1.8,
      fluoride: metrics.gender === 'male' ? 4 : 3,
      chromium: metrics.gender === 'male' ? 35 : 25,
      molybdenum: 45,
    };

    const aminoAcids = {
      histidine: Number((metrics.weightKg * 14).toFixed(1)),
      isoleucine: Number((metrics.weightKg * 19).toFixed(1)),
      leucine: Number((metrics.weightKg * 42).toFixed(1)),
      valine: Number((metrics.weightKg * 24).toFixed(1)),
      lysine: Number((metrics.weightKg * 38).toFixed(1)),
      methionine: Number((metrics.weightKg * 19).toFixed(1)),
      cysteine: Number((metrics.weightKg * 10).toFixed(1)),
      methionineCysteine: Number((metrics.weightKg * 19).toFixed(1)),
      phenylalanine: Number((metrics.weightKg * 33).toFixed(1)),
      tyrosine: Number((metrics.weightKg * 16).toFixed(1)),
      phenylalanineTyrosine: Number((metrics.weightKg * 33).toFixed(1)),
      threonine: Number((metrics.weightKg * 20).toFixed(1)),
      tryptophan: Number((metrics.weightKg * 5).toFixed(1)),
    };

    const fattyAcids = {
      omega3: Number((metrics.gender === 'male' ? 1.6 : 1.1).toFixed(2)),
      omega6: Number((metrics.gender === 'male' ? 17 : 12).toFixed(2)),
      saturatedFat: Number((calories * 0.1 / 9).toFixed(2)),
      monounsaturatedFat: Number((calories * 0.15 / 9).toFixed(2)),
      polyunsaturatedFat: Number((calories * 0.1 / 9).toFixed(2)),
      transFat: 2,
    };

    const core = {
      calories,
      protein: macros.protein,
      carbs: macros.carbs,
      fat: macros.fat,
      fiber: metrics.gender === 'male' ? 38 : 25,
      sugar: Math.round((calories * 0.1) / 4),
      water: getHydrationTarget(metrics),
      choline: metrics.gender === 'male' ? 550 : 425,
    };

    const targets = { ...core, ...vitamins, ...minerals, ...aminoAcids, ...fattyAcids };
    const targetModes = Object.fromEntries(Object.keys(targets).map((key) => [key, 'auto']));

    return {
      profile: metrics,
      bmr: Math.round(bmr),
      tdee: Math.round(tdee),
      targets,
      targetModes,
      groups: { vitamins, minerals, aminoAcids, fattyAcids },
      macros,
      hydrationTarget: core.water,
    };
  }, [profile]);

  const mapEntryToMeal = useCallback((e) => {
    const normalized = normalizeMealNutrition({
      ...(e.raw_nutrition || {}),
      ...(e.nutrition || {}),
      vitamins: e.vitamins,
      minerals: e.minerals,
      aminoAcids: e.aminoAcids || e.amino_acids,
      fattyAcids: e.fattyAcids || e.fatty_acids,
      calories: e.calories,
      protein: e.protein,
      carbs: e.carbs,
      fat: e.fat,
      fiber: e.fiber,
      sugar: e.sugar,
      water: e.water,
      choline: e.choline,
    });
    const ts = e.timestamp ? new Date(e.timestamp) : new Date();
    return {
      id: e.timestamp || e.id || `meal-${Math.random()}`,
      name: e.display_name || (e.food_label || '').replace(/_/g, ' '),
      calories: normalized.calories,
      protein: normalized.protein,
      carbs: normalized.carbs,
      fat: normalized.fat,
      fiber: normalized.fiber,
      sugar: normalized.sugar,
      water: normalized.water,
      choline: normalized.choline,
      vitamins: normalized.vitamins,
      minerals: normalized.minerals,
      aminoAcids: normalized.aminoAcids,
      fattyAcids: normalized.fattyAcids,
      nutrients: normalized.nutrients,
      fullNutrition: normalized.fullNutrition,
      source: e.source || 'logged',
      portionMultiplier: toNumber(e.portion_multiplier || e.portionMultiplier || 1),
      servingSize: e.serving_size || e.servingSize || null,
      imageUri: e.image_url || null,
      timestamp: ts,
      dateKey: toDateKey(ts),
    };
  }, []);

  const addMeal = useCallback((mealBase, imageUri = null) => {
    const nutrition = normalizeMealNutrition({
      ...(mealBase.nutrition || {}),
      ...(mealBase.nutrients || {}),
      vitamins: mealBase.vitamins,
      minerals: mealBase.minerals,
      aminoAcids: mealBase.aminoAcids || mealBase.amino_acids,
      fattyAcids: mealBase.fattyAcids || mealBase.fatty_acids,
      calories: mealBase.calories,
      protein: mealBase.protein,
      carbs: mealBase.carbs,
      fat: mealBase.fat,
      fiber: mealBase.fiber,
      sugar: mealBase.sugar,
      water: mealBase.water,
      choline: mealBase.choline,
    });
    const now = mealBase.timestamp ? new Date(mealBase.timestamp) : new Date();
    const meal = {
      id: mealBase.id || `meal-${Date.now()}`,
      name: mealBase.name,
      calories: nutrition.calories,
      protein: nutrition.protein,
      carbs: nutrition.carbs,
      fat: nutrition.fat,
      fiber: nutrition.fiber,
      sugar: nutrition.sugar,
      water: nutrition.water,
      choline: nutrition.choline,
      vitamins: nutrition.vitamins,
      minerals: nutrition.minerals,
      aminoAcids: nutrition.aminoAcids,
      fattyAcids: nutrition.fattyAcids,
      nutrients: nutrition.nutrients,
      fullNutrition: nutrition.fullNutrition,
      source: mealBase.source || 'logged',
      portionMultiplier: toNumber(mealBase.portionMultiplier ?? mealBase.portion_multiplier ?? 1),
      servingSize: mealBase.servingSize || mealBase.serving_size || null,
      imageUri: imageUri || null,
      timestamp: now,
      dateKey: toDateKey(now),
    };
    setMeals((prev) => [meal, ...prev]);
    return meal;
  }, []);

  // removeMeal: optimistically removes from local state and deletes from backend.
  // Pass token so the backend delete is authenticated.
  // If the backend call fails the meal is restored and the error is re-thrown
  // so the calling screen can show a user-visible error.
  const removeMeal = useCallback(async (mealId, token = null) => {
    // Optimistic removal
    let removed = null;
    setMeals((prev) => {
      removed = prev.find((m) => m.id === mealId) || null;
      return prev.filter((m) => m.id !== mealId);
    });

    if (!token) return; // local-only removal when not authenticated

    try {
      await apiRequest(`/map/log/${encodeURIComponent(mealId)}`, {
        method: 'DELETE',
        token,
      });
    } catch {
      // Restore the meal if backend delete failed
      if (removed) {
        setMeals((prev) => [removed, ...prev].sort((a, b) => b.timestamp - a.timestamp));
      }
      throw new Error('Could not delete meal. Please try again.');
    }
  }, []);

  // Hydrate meals from backend entries (replaces all meals with server state)
  const loadMeals = useCallback((entries) => {
    const mapped = entries.map(mapEntryToMeal);
    setMeals(mapped);
  }, [mapEntryToMeal]);

  const upsertMeals = useCallback((entries) => {
    const mapped = entries.map(mapEntryToMeal);
    setMeals((prev) => {
      const nextById = new Map(prev.map((m) => [m.id, m]));
      mapped.forEach((m) => nextById.set(m.id, m));
      return Array.from(nextById.values()).sort((a, b) => b.timestamp - a.timestamp);
    });
  }, [mapEntryToMeal]);

  const getMealsForDate = useCallback(
    (dateKey) => meals.filter((m) => m.dateKey === dateKey),
    [meals],
  );

  const getTotalsForDate = useCallback(
    (dateKey) => {
      const seed = Object.fromEntries([...CORE_NUTRIENT_KEYS, ...VITAMIN_KEYS, ...MINERAL_KEYS, ...AMINO_ACID_KEYS, ...FATTY_ACID_KEYS].map((key) => [key, 0]));
      const reduced = meals
        .filter((m) => m.dateKey === dateKey)
        .reduce((sum, meal) => {
          const source = {
            ...(meal.nutrients || {}),
            ...(meal.fullNutrition?.nutrients || {}),
            calories: meal.calories,
            protein: meal.protein,
            carbs: meal.carbs,
            fat: meal.fat,
            fiber: meal.fiber,
            sugar: meal.sugar,
            water: meal.water,
            choline: meal.choline,
            ...(meal.vitamins || {}),
            ...(meal.minerals || {}),
            ...(meal.aminoAcids || {}),
            ...(meal.fattyAcids || {}),
          };

          Object.keys(seed).forEach((key) => {
            sum[key] += toNumber(source[key]);
          });
          return sum;
        }, seed);

      // Add hydration and drink nutrients
      const hydrationTotal = hydrationLogs
        .filter((h) => h.dateKey === dateKey)
        .reduce((sum, h) => sum + h.amountMl, 0);
      
      const hydrationNutrients = {};
      hydrationLogs
        .filter((h) => h.dateKey === dateKey)
        .forEach((h) => {
          Object.entries(h.nutrients || {}).forEach(([key, value]) => {
            hydrationNutrients[key] = (hydrationNutrients[key] || 0) + value;
          });
        });

      const totalsWithHydration = {
        ...reduced,
        water: reduced.water + hydrationTotal,
        ...hydrationNutrients,
      };

      return {
        ...totalsWithHydration,
        vitamins: Object.fromEntries(VITAMIN_KEYS.map((key) => [key, totalsWithHydration[key]])),
        minerals: Object.fromEntries(MINERAL_KEYS.map((key) => [key, totalsWithHydration[key]])),
        aminoAcids: Object.fromEntries(AMINO_ACID_KEYS.map((key) => [key, totalsWithHydration[key]])),
        fattyAcids: Object.fromEntries(FATTY_ACID_KEYS.map((key) => [key, totalsWithHydration[key]])),
      };
    },
    [meals, hydrationLogs],
  );

  const getTargetComparisonForDate = useCallback(
    (dateKey, targetBundle = personalizedTargets) => {
      const totals = getTotalsForDate(dateKey);
      const targets = targetBundle?.targets || {};
      const targetModes = targetBundle?.targetModes || {};
      const allKeys = Object.keys(targets);

      const byNutrient = Object.fromEntries(
        allKeys.map((key) => {
          const state = buildTargetStatus(totals[key], targets[key]);
          return [
            key,
            {
              ...state,
              targetState: String(targetModes[key] || 'auto').toUpperCase(),
            },
          ];
        })
      );

      return {
        byNutrient,
        totals,
      };
    },
    [getTotalsForDate, personalizedTargets],
  );

  const getAchievementMetrics = useCallback(() => {
    const target = personalizedTargets?.targets || {};
    const dayKeySet = new Set();
    meals.forEach((m) => dayKeySet.add(m.dateKey));
    hydrationLogs.forEach((h) => dayKeySet.add(h.dateKey));
    const dayKeys = sortDateKeysAsc(Array.from(dayKeySet));

    let calorieTargetHits = 0;
    let waterTargetHits = 0;
    let proteinTargetHits = 0;
    let macroGoalsHits = 0;
    let fiberGoalHits = 0;
    let vitaminMineralPerfectDays = 0;

    dayKeys.forEach((dateKey) => {
      const totals = getTotalsForDate(dateKey);
      const caloriesRatio = toNumber(target.calories) > 0 ? toNumber(totals.calories) / toNumber(target.calories) : 0;
      if (caloriesRatio >= 0.97 && caloriesRatio <= 1.03) calorieTargetHits += 1;

      if (toNumber(target.water) > 0 && toNumber(totals.water) >= toNumber(target.water)) waterTargetHits += 1;
      if (toNumber(target.protein) > 0 && toNumber(totals.protein) >= toNumber(target.protein)) proteinTargetHits += 1;
      if (toNumber(target.fiber) > 0 && toNumber(totals.fiber) >= toNumber(target.fiber)) fiberGoalHits += 1;

      const macroKeys = ['protein', 'carbs', 'fat'];
      const macrosHit = macroKeys.every((key) => {
        const t = toNumber(target[key]);
        if (t <= 0) return false;
        const ratio = toNumber(totals[key]) / t;
        return ratio >= 0.95 && ratio <= 1.05;
      });
      if (macrosHit) macroGoalsHits += 1;

      const vitaminOk = VITAMIN_KEYS.every((key) => {
        const t = toNumber(target[key]);
        if (t <= 0) return true;
        return toNumber(totals[key]) / t >= 1;
      });
      const mineralOk = MINERAL_KEYS.every((key) => {
        const t = toNumber(target[key]);
        if (t <= 0) return true;
        return toNumber(totals[key]) / t >= 1;
      });
      if (vitaminOk && mineralOk) vitaminMineralPerfectDays += 1;
    });

    const mealDateKeys = sortDateKeysAsc(Array.from(new Set(meals.map((m) => m.dateKey))));
    const hydrationDateKeys = sortDateKeysAsc(
      Array.from(new Set(hydrationLogs.filter((h) => toNumber(target.water) > 0).filter((h) => {
        const totals = getTotalsForDate(h.dateKey);
        return toNumber(totals.water) >= toNumber(target.water);
      }).map((h) => h.dateKey))),
    );

    const computeCurrentStreak = (keys) => {
      if (!keys.length) return 0;
      const keySet = new Set(keys);
      let streak = 0;
      let cursor = toDateKey(new Date());
      while (keySet.has(cursor)) {
        streak += 1;
        cursor = previousDateKey(cursor);
      }
      return streak;
    };

    return {
      firstMealLogged: meals.length > 0,
      totalMealsLogged: meals.length,
      daysWithMeals: mealDateKeys.length,
      currentMealStreak: computeCurrentStreak(mealDateKeys),
      currentHydrationStreak: computeCurrentStreak(hydrationDateKeys),
      calorieTargetHits,
      waterTargetHits,
      proteinTargetHits,
      macroGoalsHits,
      fiberGoalHits,
      vitaminMineralPerfectDays,
      favoritesSavedCount: favoriteMealIds.size,
      scansCompleted: meals.filter((m) => m.source === 'logged' || m.source === 'saved').length,
      correctionsSubmitted: correctionSubmissions,
    };
  }, [correctionSubmissions, favoriteMealIds, getTotalsForDate, hydrationLogs, meals, personalizedTargets]);

  const getSummaryForDate = useCallback(
    (dateKey, calorieGoal = null) => {
      const dayMeals = meals.filter((m) => m.dateKey === dateKey);
      if (dayMeals.length === 0) return null;
      const consumed = getTotalsForDate(dateKey).calories;
      const goal = Number.isFinite(Number(calorieGoal)) && Number(calorieGoal) > 0
        ? Number(calorieGoal)
        : toNumber(personalizedTargets?.targets?.calories);
      const ratio = goal > 0 ? consumed / goal : 0;
      let status = 'low';
      if (ratio > 1.05) status = 'over';
      else if (ratio >= 0.75) status = 'on-track';
      return {
        consumed,
        goal,
        remaining: Math.max(0, goal - consumed),
        percentReached: goal > 0 ? Math.round((consumed / goal) * 100) : 0,
        status,
        mealCount: dayMeals.length,
      };
    },
    [getTotalsForDate, meals, personalizedTargets],
  );

  const awardInstantXp = useCallback((amount, source = 'general', metadata = {}) => {
    const xp = Math.max(0, Number(amount) || 0);
    if (xp <= 0) return null;
    const now = new Date();
    const event = {
      id: `xp-${Date.now()}`,
      source,
      amount: xp,
      dateKey: toDateKey(now),
      timestamp: now,
      ...metadata,
    };
    setInstantXpEvents((prev) => [event, ...prev]);
    return event;
  }, []);

  const registerCorrectionSubmission = useCallback((xpReward = 100) => {
    setCorrectionSubmissions((prev) => prev + 1);
    return awardInstantXp(xpReward, 'correction', { kind: 'wrong-prediction-fix' });
  }, [awardInstantXp]);

  const getDailyXpForDate = useCallback(
    (dateKey) => {
      const totals = getTotalsForDate(dateKey);
      const targets = personalizedTargets?.targets || {};
      const goalType = personalizedTargets?.profile?.goalType || 'maintain';
      const dayMeals = meals.filter((m) => m.dateKey === dateKey);
      const hydrationForDay = hydrationLogs.filter((h) => h.dateKey === dateKey);
      const macroWeights = GOAL_MACRO_XP_WEIGHTS[goalType] || GOAL_MACRO_XP_WEIGHTS.maintain;
      const reasons = [];

      const calorieScore = scoreCalorieByGoal(totals.calories, targets.calories, goalType);
      const caloriesXp = roundXp(calorieScore.points);
      reasons.push(`${caloriesXp >= 70 ? '+' : '-'}${Math.round(caloriesXp)} XP calories: ${calorieScore.reason}`);

      const macroKeys = ['protein', 'carbs', 'fat'];
      const macroDetails = macroKeys.map((key) => {
        const consumed = toNumber(totals[key]);
        const target = toNumber(targets[key]);
        const ratio = target > 0 ? consumed / target : 0;
        let multiplier = ratioScoreBand(ratio);

        if (goalType === 'lose') {
          if (key === 'carbs' && ratio > 1.1) multiplier *= 0.6;
          if (key === 'protein' && ratio < 0.9) multiplier *= 0.45;
          if (key === 'fat' && ratio > 1.2) multiplier *= 0.6;
        }

        if (goalType === 'muscle') {
          if (key === 'protein' && ratio < 0.85) multiplier = 0;
          else if (key === 'protein' && ratio < 0.95) multiplier *= 0.4;
          if (key === 'carbs' && ratio < 0.85) multiplier *= 0.7;
          if (key === 'fat' && (ratio < 0.7 || ratio > 1.35)) multiplier *= 0.75;
        }

        if (goalType === 'gain') {
          if ((key === 'protein' || key === 'carbs') && ratio < 0.9) multiplier *= 0.6;
          if (key === 'fat' && ratio > 1.3) multiplier *= 0.75;
        }

        const bucket = toNumber(macroWeights[key]);
        const earned = roundXp(bucket * clamp(multiplier, 0, 1));
        return {
          key,
          target,
          consumed,
          ratio: Number(ratio.toFixed(3)),
          maxXp: bucket,
          earned,
        };
      });

      const macrosXp = roundXp(macroDetails.reduce((sum, item) => sum + item.earned, 0));
      macroDetails.forEach((macro) => {
        const label = macro.key.charAt(0).toUpperCase() + macro.key.slice(1);
        if (macro.earned >= macro.maxXp * 0.95) reasons.push(`+${Math.round(macro.earned)} XP for hitting ${label} target`);
        else reasons.push(`-${Math.round(macro.maxXp - macro.earned)} XP: ${label} is off target for ${goalType} goal`);
      });

      const fiberRatio = toNumber(targets.fiber) > 0 ? toNumber(totals.fiber) / toNumber(targets.fiber) : 0;
      const fiberMultiplier = fiberRatio >= 1 ? 1 : fiberRatio >= 0.8 ? 0.75 : fiberRatio >= 0.6 ? 0.4 : 0.1;
      const fiberXp = roundXp(XP_BUCKETS.fiber * fiberMultiplier);

      const cholineRatio = toNumber(targets.choline) > 0 ? toNumber(totals.choline) / toNumber(targets.choline) : 0;
      const cholineMultiplier = cholineRatio >= 1 ? 1 : cholineRatio >= 0.8 ? 0.75 : cholineRatio >= 0.6 ? 0.4 : 0.1;
      const cholineXp = roundXp(XP_BUCKETS.choline * cholineMultiplier);

      const waterRatio = toNumber(targets.water) > 0 ? toNumber(totals.water) / toNumber(targets.water) : 0;
      const waterMultiplier = waterRatio >= 1 ? 1 : waterRatio >= 0.8 ? 0.75 : waterRatio >= 0.6 ? 0.4 : 0.1;
      const waterXp = roundXp(XP_BUCKETS.water * waterMultiplier);

      const vitaminScore = scoreWeightedNutrients({
        totals,
        targets,
        weights: VITAMIN_XP_WEIGHTS,
        keys: Object.keys(VITAMIN_XP_WEIGHTS),
        bandFn: ratioScoreBandMicronutrient,
        goalType,
      });
      const vitaminsXp = roundXp(vitaminScore.points);

      const majorMineralScore = scoreWeightedNutrients({
        totals,
        targets,
        weights: MAJOR_MINERAL_XP_WEIGHTS,
        keys: MAJOR_MINERAL_KEYS,
        bandFn: ratioScoreBandMicronutrient,
        goalType,
      });
      const majorMineralsXp = roundXp(majorMineralScore.points);

      const traceMineralScore = scoreWeightedNutrients({
        totals,
        targets,
        weights: TRACE_MINERAL_XP_WEIGHTS,
        keys: TRACE_MINERAL_KEYS,
        bandFn: ratioScoreBandMicronutrient,
        goalType,
      });
      const traceMineralsXp = roundXp(traceMineralScore.points);

      const aminoScore = scoreWeightedNutrients({
        totals,
        targets,
        weights: AMINO_XP_WEIGHTS,
        keys: ESSENTIAL_AMINO_KEYS,
        bandFn: ratioScoreBandMicronutrient,
        goalType,
        groupType: 'amino',
      });
      const aminoAcidsXp = roundXp(aminoScore.points);

      const fattyScore = scoreWeightedNutrients({
        totals,
        targets,
        weights: FATTY_XP_WEIGHTS,
        keys: Object.keys(FATTY_XP_WEIGHTS),
        bandFn: ratioScoreBandMicronutrient,
        goalType,
      });
      const fattyAcidsXp = roundXp(fattyScore.points);

      reasons.push(`${fiberXp >= 18 ? '+' : '-'}${Math.round(fiberXp)} XP fiber target`);
      reasons.push(`${cholineXp >= 14 ? '+' : '-'}${Math.round(cholineXp)} XP choline target`);
      reasons.push(`${waterXp >= 18 ? '+' : '-'}${Math.round(waterXp)} XP hydration target`);

      let bonusXp = 0;
      if (dayMeals.length >= 3) {
        bonusXp += 8;
        reasons.push('+8 XP bonus: all meals logged');
      }
      if (hydrationForDay.length >= 3 && waterRatio >= 1) {
        bonusXp += 7;
        reasons.push('+7 XP bonus: hydration consistency');
      }
      if (caloriesXp >= 75) {
        bonusXp += 5;
        reasons.push('+5 XP bonus: stayed near calorie target');
      }
      if (macroDetails.every((m) => m.earned >= m.maxXp * 0.75) && fiberXp >= 15 && waterXp >= 15) {
        bonusXp += 10;
        reasons.push('+10 XP bonus: strong major target coverage');
      }
      bonusXp = roundXp(clamp(bonusXp, 0, XP_BUCKETS.bonus));

      const breakdown = {
        calories: caloriesXp,
        macros: macrosXp,
        fiber: fiberXp,
        choline: cholineXp,
        water: waterXp,
        vitamins: vitaminsXp,
        majorMinerals: majorMineralsXp,
        traceMinerals: traceMineralsXp,
        aminoAcids: aminoAcidsXp,
        fattyAcids: fattyAcidsXp,
        bonus: bonusXp,
      };

      const maxCoreXp = XP_BUCKETS.calories + XP_BUCKETS.macros + XP_BUCKETS.fiber + XP_BUCKETS.choline + XP_BUCKETS.water + XP_BUCKETS.vitamins + XP_BUCKETS.majorMinerals + XP_BUCKETS.traceMinerals + XP_BUCKETS.aminoAcids + XP_BUCKETS.fattyAcids;
      const totalXp = roundXp(Object.values(breakdown).reduce((sum, value) => sum + value, 0));
      const nutritionScore = roundXp((Math.min(totalXp - bonusXp, maxCoreXp) / maxCoreXp) * 100);
      const goalAlignmentBase = caloriesXp + macrosXp;
      const goalAlignmentMax = XP_BUCKETS.calories + XP_BUCKETS.macros;
      const goalAlignmentScore = roundXp((goalAlignmentBase / goalAlignmentMax) * 100);

      return {
        dateKey,
        goalType,
        totalXp,
        nutritionScore,
        goalAlignmentScore,
        breakdown,
        maxByCategory: XP_BUCKETS,
        macroBreakdown: macroDetails,
        vitaminsBreakdown: vitaminScore.details,
        majorMineralsBreakdown: majorMineralScore.details,
        traceMineralsBreakdown: traceMineralScore.details,
        aminoAcidsBreakdown: aminoScore.details,
        fattyAcidsBreakdown: fattyScore.details,
        reasons,
      };
    },
    [getTotalsForDate, personalizedTargets, meals, hydrationLogs],
  );

  const getXpProgression = useCallback(
    () => {
      const dateSet = new Set();
      meals.forEach((meal) => dateSet.add(meal.dateKey));
      hydrationLogs.forEach((h) => dateSet.add(h.dateKey));

      const orderedKeys = sortDateKeysAsc(Array.from(dateSet));
      const daily = orderedKeys.map((dateKey) => getDailyXpForDate(dateKey));
      const achievementXp = roundXp(
        (achievements || []).reduce((sum, item) => {
          const status = String(item?.status || '').toLowerCase();
          const isClaimed = status === 'claimed' || status === 'completed';
          return sum + (isClaimed ? toNumber(item?.xp_reward || item?.xpReward) : 0);
        }, 0),
      );
      const instantXp = roundXp(instantXpEvents.reduce((sum, event) => sum + toNumber(event?.amount), 0));
      const totalXp = roundXp(daily.reduce((sum, day) => sum + day.totalXp, 0) + achievementXp + instantXp);
      const level = Math.max(1, Math.floor(totalXp / XP_LEVEL_STEP) + 1);
      const xpIntoLevel = roundXp(totalXp % XP_LEVEL_STEP);
      const xpToNextLevel = roundXp(XP_LEVEL_STEP - xpIntoLevel);

      const todayKey = toDateKey(new Date());
      const dailyMap = new Map(daily.map((d) => [d.dateKey, d]));
      let streak = 0;
      let cursor = todayKey;
      while (dailyMap.has(cursor) && dailyMap.get(cursor).totalXp >= STRONG_DAY_XP) {
        streak += 1;
        cursor = previousDateKey(cursor);
      }

      const lastSeven = daily
        .filter((d) => new Date(d.dateKey) >= new Date(Date.now() - 6 * 24 * 60 * 60 * 1000));
      const weeklyConsistencyXp = roundXp(lastSeven.reduce((sum, day) => sum + day.totalXp, 0));

      const badges = [];
      if (streak >= 3) badges.push('3-day nutrition streak');
      if (streak >= 7) badges.push('7-day nutrition streak');
      if (daily.some((d) => d.nutritionScore >= 90)) badges.push('High precision day');
      if (daily.some((d) => d.breakdown.water >= XP_BUCKETS.water)) badges.push('Hydration finisher');

      return {
        totalXp,
        achievementXp,
        instantXp,
        level,
        xpIntoLevel,
        xpToNextLevel,
        streak,
        weeklyConsistencyXp,
        badges,
        daily,
      };
    },
    [achievements, getDailyXpForDate, meals, hydrationLogs, instantXpEvents],
  );

  // Set of dateKeys that have at least one meal — used for calendar dot indicators
  const activeDateKeys = React.useMemo(() => {
    const keys = new Set();
    meals.forEach((m) => keys.add(m.dateKey));
    return keys;
  }, [meals]);

  // Save a meal (from logged meal) as a reusable template
  const saveMealTemplate = useCallback((meal, saveName = null) => {
    if (!meal) return null;
    const createdAt = new Date();
    const userId = profile?.id || profile?.user_id || profile?.userId || null;
    const savedMeal = {
      templateId: `template-${Date.now()}`,
      user_id: userId,
      name: saveName || meal.name,
      calories: meal.calories,
      protein: meal.protein,
      carbs: meal.carbs,
      fat: meal.fat,
      macros: {
        protein: meal.protein,
        carbs: meal.carbs,
        fat: meal.fat,
      },
      fiber: meal.fiber,
      sugar: meal.sugar,
      water: meal.water,
      choline: meal.choline,
      vitamins: meal.vitamins || {},
      minerals: meal.minerals || {},
      micronutrients: {
        vitamins: meal.vitamins || {},
        minerals: meal.minerals || {},
      },
      aminoAcids: meal.aminoAcids || {},
      fattyAcids: meal.fattyAcids || {},
      nutrients: meal.nutrients || {},
      fullNutrition: meal.fullNutrition,
      imageUri: meal.imageUri || null,
      servingSize: meal.servingSize || null,
      sourceId: meal.id || null,
      createdAt,
      created_at: createdAt,
      lastUsedAt: null,
      useCount: 0,
    };
    setSavedMeals((prev) => [savedMeal, ...prev]);
    return savedMeal;
  }, [profile]);

  // Toggle favorite status for a saved meal
  const toggleFavoriteMeal = useCallback((templateId) => {
    setFavoriteMealIds((prev) => {
      const next = new Set(prev);
      if (next.has(templateId)) {
        next.delete(templateId);
      } else {
        next.add(templateId);
      }
      return next;
    });
  }, []);

  // Get favorite meals
  const getFavoriteMeals = useCallback(() => {
    return savedMeals.filter((m) => favoriteMealIds.has(m.templateId));
  }, [savedMeals, favoriteMealIds]);

  // Get recent meals (recently used saved meals, sorted by lastUsedAt)
  const getRecentMeals = useCallback((limit = 5) => {
    return savedMeals
      .filter((m) => m.lastUsedAt)
      .sort((a, b) => new Date(b.lastUsedAt) - new Date(a.lastUsedAt))
      .slice(0, limit);
  }, [savedMeals]);

  // Delete a saved meal template
  const deleteSavedMeal = useCallback((templateId) => {
    setSavedMeals((prev) => prev.filter((m) => m.templateId !== templateId));
    setFavoriteMealIds((prev) => {
      const next = new Set(prev);
      next.delete(templateId);
      return next;
    });
  }, []);

  // Log a saved meal with optional portion adjustment
  const logSavedMeal = useCallback((savedMeal, portionMultiplier = 1) => {
    if (!savedMeal) return null;
    // Scale all nutrition values by portion multiplier
    const scaledMeal = {
      name: savedMeal.name,
      calories: Math.round(savedMeal.calories * portionMultiplier),
      protein: Math.round(savedMeal.protein * portionMultiplier),
      carbs: Math.round(savedMeal.carbs * portionMultiplier),
      fat: Math.round(savedMeal.fat * portionMultiplier),
      fiber: Number((savedMeal.fiber * portionMultiplier).toFixed(2)),
      sugar: Number((savedMeal.sugar * portionMultiplier).toFixed(2)),
      water: Number((savedMeal.water * portionMultiplier).toFixed(2)),
      choline: Number((savedMeal.choline * portionMultiplier).toFixed(2)),
      vitamins: Object.fromEntries(
        Object.entries(savedMeal.vitamins || {}).map(([k, v]) => [k, Number((v * portionMultiplier).toFixed(2))])
      ),
      minerals: Object.fromEntries(
        Object.entries(savedMeal.minerals || {}).map(([k, v]) => [k, Number((v * portionMultiplier).toFixed(2))])
      ),
      aminoAcids: Object.fromEntries(
        Object.entries(savedMeal.aminoAcids || {}).map(([k, v]) => [k, Number((v * portionMultiplier).toFixed(2))])
      ),
      fattyAcids: Object.fromEntries(
        Object.entries(savedMeal.fattyAcids || {}).map(([k, v]) => [k, Number((v * portionMultiplier).toFixed(2))])
      ),
      source: 'saved',
      portionMultiplier,
      servingSize: savedMeal.servingSize,
    };
    const loggedMeal = addMeal(scaledMeal, savedMeal.imageUri);
    // Update lastUsedAt and useCount for the saved meal
    setSavedMeals((prev) =>
      prev.map((m) =>
        m.templateId === savedMeal.templateId
          ? { ...m, lastUsedAt: new Date(), useCount: (m.useCount || 0) + 1 }
          : m
      )
    );
    return loggedMeal;
  }, [addMeal]);

  // Add hydration entry (water or flavored drink)
  const logHydration = useCallback((amount, unit = 'ml', drinkType = 'plain', addedNutrients = {}) => {
    const amountMl = convertToMl(amount, unit);
    const drinkDef = DRINK_TYPES[drinkType] || DRINK_TYPES.plain;
    const now = new Date();
    const dateKey = toDateKey(now);

    // Scale nutrients based on amount (drinkType nutrients are for 250ml)
    const scaleFactor = amountMl / 250;
    const nutrients = Object.fromEntries(
      Object.entries(drinkDef.nutrients || {}).map(([key, value]) => [key, Number((value * scaleFactor).toFixed(2))])
    );

    const hydrationEntry = {
      id: `hydration-${Date.now()}`,
      timestamp: now,
      dateKey,
      drinkType,
      amount,
      unit,
      amountMl,
      nutrients,
      caffeine: drinkDef.caffeine ? Math.round(drinkDef.caffeine * scaleFactor) : null,
      ...addedNutrients,
    };

    setHydrationLogs((prev) => [hydrationEntry, ...prev]);
    return hydrationEntry;
  }, []);

  // Get total hydration for a specific date
  const getHydrationForDate = useCallback((dateKey) => {
    return hydrationLogs
      .filter((h) => h.dateKey === dateKey)
      .reduce((sum, h) => sum + h.amountMl, 0);
  }, [hydrationLogs]);

  // Get all nutrients from hydration for a date (minerals, electrolytes, etc.)
  const getHydrationNutrientsForDate = useCallback((dateKey) => {
    const result = {};
    hydrationLogs
      .filter((h) => h.dateKey === dateKey)
      .forEach((h) => {
        Object.entries(h.nutrients || {}).forEach(([key, value]) => {
          result[key] = (result[key] || 0) + value;
        });
      });
    return result;
  }, [hydrationLogs]);

  // Remove hydration entry
  const removeHydration = useCallback((hydrationId) => {
    setHydrationLogs((prev) => prev.filter((h) => h.id !== hydrationId));
  }, []);

  return (
    <MealContext.Provider value={{
      meals,
      addMeal,
      removeMeal,
      loadMeals,
      upsertMeals,
      getMealsForDate,
      getTotalsForDate,
      getSummaryForDate,
      getTargetComparisonForDate,
      getDailyXpForDate,
      getXpProgression,
      getAchievementMetrics,
      awardInstantXp,
      registerCorrectionSubmission,
      personalizedTargets,
      activeDateKeys,
      savedMeals,
      saveMealTemplate,
      toggleFavoriteMeal,
      getFavoriteMeals,
      getRecentMeals,
      deleteSavedMeal,
      logSavedMeal,
      favoriteMealIds,
      hydrationLogs,
      logHydration,
      getHydrationForDate,
      getHydrationNutrientsForDate,
      removeHydration,
      instantXpEvents,
      correctionSubmissions,
    }}>
      {children}
    </MealContext.Provider>
  );
}

export function useMeals() {
  const ctx = useContext(MealContext);
  if (!ctx) throw new Error('useMeals must be used inside MealProvider');
  return ctx;
}

export { toDateKey, DRINK_TYPES, convertToMl, UNIT_TO_ML, getHydrationTarget };
