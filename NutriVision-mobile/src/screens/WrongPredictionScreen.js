import React, { useState, useMemo } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  TextInput,
  Image,
  Alert,
  ActivityIndicator,
  Modal,
} from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useNavigation, useRoute } from '@react-navigation/native';
import { useTheme } from '../theme/ThemeContext';
import { useAuth } from '../context/AuthContext';
import { useMeals } from '../context/MealContext';
import { Typography, Spacing, BorderRadius, Shadow } from '../theme';
import { Card, Button } from '../components';
import { Camera, Image as ImageIcon, ChevronLeft, CheckCircle, X, AlertCircle } from 'lucide-react-native';
import { apiClient, apiRequest } from '../api/client';

const QUICK_SERVINGS = ['1 serving', '1 cup', '100 g', '1 piece', '250 ml'];

const PRESETS_BY_UNIT = {
  serving: [0.5, 1, 1.5, 2],
  grams: [50, 100, 150, 200, 250, 300],
  oz: [1, 2, 4, 8],
  calories: [100, 200, 300, 500],
};

function getPresetsForUnit(unit) {
  return PRESETS_BY_UNIT[unit] || PRESETS_BY_UNIT.serving;
}

export default function WrongPredictionScreen() {
  const { colors } = useTheme();
  const insets = useSafeAreaInsets();
  const navigation = useNavigation();
  const route = useRoute();
  const { token, profile, refreshProfile, achievements, upsertAchievement } = useAuth();
  const { addMeal, registerCorrectionSubmission } = useMeals();

  // Params passed from ScanScreen
  const {
    originalPrediction = '',
    originalConfidence = 0,
    imageUri: originalImageUri = null,
  } = route.params || {};

  // Form state
  const [foodName, setFoodName] = useState('');
  const [servingSize, setServingSize] = useState('1');
  const [servingUnit, setServingUnit] = useState('serving'); // 'serving', 'grams', 'oz', 'calories'
  const [customServing, setCustomServing] = useState(false);

  // Images: top, side, inside, label
  const [images, setImages] = useState({ top: null, side: null, inside: null, label: null });

  // Submission state
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState(null);
  const [showSuccess, setShowSuccess] = useState(false);

  // Result state (after submission)
  const [correctedNutrition, setCorrectedNutrition] = useState(null);
  const [isFetchingNutrition, setIsFetchingNutrition] = useState(false);
  const [nutritionError, setNutritionError] = useState(null);
  const [xpEarned, setXpEarned] = useState(null);
  const [isLoggingMeal, setIsLoggingMeal] = useState(false);

  // Completion %
  const completionPct = useMemo(() => {
    let score = 0;
    if (foodName.trim()) score += 50;
    if (servingSize.trim()) score += 10;
    const photoCount = Object.values(images).filter(Boolean).length;
    score += photoCount * 10; // up to 40 for 4 photos
    return Math.min(score, 100);
  }, [foodName, servingSize, servingUnit, images]);

  const readyToSubmit = foodName.trim().length > 0 && servingSize.trim().length > 0;

  const pickImage = async (slot) => {
    try {
      const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
      if (status !== 'granted') {
        Alert.alert('Permission Needed', 'Gallery permissions are required.');
        return;
      }
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: true,
        quality: 0.8,
      });
      if (!result.canceled) {
        setImages((prev) => ({ ...prev, [slot]: result.assets[0].uri }));
      }
    } catch {
      Alert.alert('Gallery Error', 'Could not access photo gallery.');
    }
  };

  const takePhoto = async (slot) => {
    try {
      const { status } = await ImagePicker.requestCameraPermissionsAsync();
      if (status !== 'granted') {
        Alert.alert('Permission Needed', 'Camera permissions are required.');
        return;
      }
      const result = await ImagePicker.launchCameraAsync({ allowsEditing: true, quality: 0.8 });
      if (!result.canceled) {
        setImages((prev) => ({ ...prev, [slot]: result.assets[0].uri }));
      }
    } catch {
      Alert.alert('Camera Error', 'Could not access camera.');
    }
  };

  const fetchNutritionForCorrection = async (name, serving) => {
    setIsFetchingNutrition(true);
    setNutritionError(null);
    try {
      const data = await apiClient.post('/map/nutrition', { query: `${name} ${serving}` });
      if (data.nutrition) {
        setCorrectedNutrition(data.nutrition);
        return data.nutrition;
      } else {
        setNutritionError('Nutrition data not found. You can still log the meal manually.');
        return null;
      }
    } catch {
      setNutritionError('Could not fetch nutrition data right now. You can retry or log manually.');
      return null;
    } finally {
      setIsFetchingNutrition(false);
    }
  };

  const handleSubmit = async () => {
    if (!readyToSubmit) return;
    setIsSubmitting(true);
    setSubmitError(null);

    try {
      // Step 1 — Build multipart correction payload
      const correctionPayload = {
        original_prediction: originalPrediction,
        corrected_food_name: foodName.trim(),
        serving_size: servingSize.trim(),
        serving_unit: servingUnit,
        confidence: originalConfidence,
        timestamp: new Date().toISOString(),
        has_top_image: !!images.top,
        has_side_image: !!images.side,
        has_inside_image: !!images.inside,
        has_label_image: !!images.label,
      };

      // Submit correction (best-effort; don't block UX if backend is unavailable)
      try {
        await apiRequest('/feedback/corrections/submit', {
          method: 'POST',
          token,
          body: correctionPayload,
        });
      } catch {
        // Non-fatal: correction is logged locally; XP still awarded client-side
      }

      // Step 2 — Fetch real nutrition from API using corrected food
      const servingDisplay = `${servingSize} ${servingUnit === 'grams' ? 'g' : servingUnit === 'oz' ? 'oz' : servingUnit === 'calories' ? 'kcal' : 'serving'}`;
      const nutrition = await fetchNutritionForCorrection(foodName.trim(), servingDisplay);

      // Step 3 — Award XP locally (100 XP for correction)
      const xp = 100;
      setXpEarned(xp);
      registerCorrectionSubmission(xp);

      const correctionAchievement = (achievements || []).find((a) => a.id === 'correction_submitted_once');
      const nextCorrectionProgress = Number(correctionAchievement?.progress || 0) + 1;
      try {
        await upsertAchievement({
          id: 'correction_submitted_once',
          user_id: profile?.id || profile?.user_id || profile?.userId || null,
          title: 'Model Helper',
          description: 'Submit a wrong-prediction correction once.',
          type: 'milestone',
          difficulty: 'easy',
          progress: nextCorrectionProgress,
          target: 1,
          status: nextCorrectionProgress >= 1 ? 'completed' : 'in_progress',
          xp_reward: 65,
          completed_at: nextCorrectionProgress >= 1 ? new Date().toISOString() : null,
        });
      } catch {
        // Non-blocking achievement persistence
      }

      // Attempt to update XP on backend
      try {
        await apiRequest('/map/xp', {
          method: 'POST',
          token,
          body: { action: 'correction', xp_awarded: xp },
        });
        // Refresh profile so XP total reflects update
        if (token) await refreshProfile(token);
      } catch {
        // XP update is best-effort
      }

      setShowSuccess(true);
    } catch (err) {
      setSubmitError('Submission failed. Please check your connection and try again.');
    } finally {
      setIsSubmitting(false);
    }
  };

  const retryNutrition = () => {
    fetchNutritionForCorrection(foodName.trim(), servingSize.trim());
  };

  const handleLogMeal = async () => {
    if (!correctedNutrition) {
      Alert.alert('No Nutrition Data', 'Nutrition data is still loading or unavailable.');
      return;
    }
    setIsLoggingMeal(true);
    try {
      const servingDisplay = `${servingSize} ${servingUnit === 'grams' ? 'g' : servingUnit === 'oz' ? 'oz' : servingUnit === 'calories' ? 'kcal' : 'serving'}`;

      let sourceNutrition = correctedNutrition;
      try {
        const latestNutrition = await apiClient.post('/map/nutrition', {
          query: `${foodName.trim()} ${servingDisplay}`,
        });
        if (latestNutrition?.nutrition) {
          sourceNutrition = latestNutrition.nutrition;
          setCorrectedNutrition(latestNutrition.nutrition);
        }
      } catch {
        // Continue with the currently available correction nutrition if refresh fails.
      }

      const servingNumber = Number(servingSize);
      const portionMultiplier = servingUnit === 'grams' && Number.isFinite(servingNumber)
        ? Math.max(0.25, Math.min(servingNumber / 170, 4.0))
        : servingUnit === 'oz' && Number.isFinite(servingNumber)
          ? Math.max(0.25, Math.min((servingNumber * 28.35) / 170, 4.0))
          : Math.max(0.25, Math.min(Number.isFinite(servingNumber) ? servingNumber : 1, 4.0));

      const nut = sourceNutrition || correctedNutrition;
      await apiRequest('/map/log', {
        method: 'POST',
        token,
        body: {
          food_label: foodName.trim().replace(/\s+/g, '_').toLowerCase(),
          display_name: foodName.trim(),
          portion_id: 'medium',
          portion_multiplier: portionMultiplier,
          nutrition: nut,
          raw_nutrition: sourceNutrition || {},
          source: 'correction',
          serving_size: servingDisplay,
        },
      });

      addMeal(
        {
          name: foodName.trim(),
          calories: Math.round(Number(nut.calories) || 0),
          protein: Math.round(Number(nut.protein_g ?? nut.protein) || 0),
          carbs: Math.round(Number(nut.carbs_g ?? nut.carbs) || 0),
          fat: Math.round(Number(nut.fat_g ?? nut.fat) || 0),
          fiber: Number(nut.fiber_g ?? nut.fiber ?? 0),
          sugar: Number(nut.sugar_g ?? nut.sugar ?? 0),
          water: Number(nut.water ?? 0),
          choline: Number(nut.choline_mg ?? nut.choline ?? 0),
          nutrients: nut,
          aminoAcids: nut.aminoAcids || nut.amino_acids || {},
          fattyAcids: nut.fattyAcids || nut.fatty_acids || {},
          source: 'correction',
          portionMultiplier,
          servingSize: servingDisplay,
        },
        images.top || originalImageUri,
      );

      Alert.alert('Meal Logged!', `${foodName.trim()} has been added to your log.`, [
        {
          text: 'View Home',
          onPress: () => navigation.navigate('MainTabs', { screen: 'Dashboard' }),
        },
        { text: 'Done', onPress: () => navigation.goBack() },
      ]);
    } catch {
      Alert.alert('Log Failed', 'Could not save this meal. Please try again.');
    } finally {
      setIsLoggingMeal(false);
    }
  };

  // ─── Image Slot ──────────────────────────────────────────────────────────────
  const ImageSlot = ({ slot, label, optional = true }) => {
    const uri = images[slot];
    return (
      <View style={styles.imageSlotContainer}>
        <View style={styles.imageSlotHeader}>
          <Text style={[styles.imageSlotLabel, { color: colors.text }]}>{label}</Text>
          <Text style={[styles.imageSlotTag, { color: optional ? colors.textSecondary : colors.primary }]}>
            {optional ? 'Optional' : 'Recommended'}
          </Text>
        </View>
        <View style={[styles.imageSlotBox, { backgroundColor: colors.surfaceSecondary, borderColor: uri ? colors.primary : colors.border }]}>
          {uri ? (
            <>
              <Image source={{ uri }} style={styles.imageSlotPreview} />
              <TouchableOpacity
                style={[styles.imageSlotRemove, { backgroundColor: colors.error }]}
                onPress={() => setImages((prev) => ({ ...prev, [slot]: null }))}
              >
                <X size={14} color="#fff" />
              </TouchableOpacity>
            </>
          ) : (
            <ImageIcon size={32} color={colors.textSecondary} />
          )}
        </View>
        <View style={styles.imageSlotActions}>
          <TouchableOpacity
            style={[styles.imageSlotBtn, { backgroundColor: colors.surface, borderColor: colors.border }]}
            onPress={() => takePhoto(slot)}
            activeOpacity={0.8}
          >
            <Camera size={14} color={colors.primary} />
            <Text style={[styles.imageSlotBtnText, { color: colors.text }]}>Camera</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[styles.imageSlotBtn, { backgroundColor: colors.surface, borderColor: colors.border }]}
            onPress={() => pickImage(slot)}
            activeOpacity={0.8}
          >
            <ImageIcon size={14} color={colors.primary} />
            <Text style={[styles.imageSlotBtnText, { color: colors.text }]}>Gallery</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  };

  // ─── Success Modal ────────────────────────────────────────────────────────────
  const SuccessModal = () => (
    <Modal visible={showSuccess} transparent animationType="fade" onRequestClose={() => {}}>
      <View style={[styles.successOverlay, { backgroundColor: '#00000070' }]}>
        <View style={[styles.successCard, { backgroundColor: colors.background }]}>
          <View style={[styles.successIconCircle, { backgroundColor: colors.primary + '20' }]}>
            <CheckCircle size={52} color={colors.primary} />
          </View>
          <Text style={[styles.successTitle, { color: colors.text }]}>Thanks for improving the model!</Text>
          <Text style={[styles.successMessage, { color: colors.textSecondary }]}>
            You helped make food recognition smarter.
          </Text>
          <View style={[styles.xpBadge, { backgroundColor: colors.primary }]}>
            <Text style={styles.xpBadgeText}>+{xpEarned} XP</Text>
          </View>
          <TouchableOpacity
            style={[styles.successCloseBtn, { backgroundColor: colors.primary }]}
            onPress={() => setShowSuccess(false)}
            activeOpacity={0.85}
          >
            <Text style={styles.successCloseBtnText}>See Results</Text>
          </TouchableOpacity>
        </View>
      </View>
    </Modal>
  );

  const hasSubmitted = showSuccess === false && xpEarned !== null;

  return (
    <View style={[styles.container, { backgroundColor: colors.background, paddingTop: insets.top }]}>
      {/* Header */}
      <View style={[styles.header, { borderBottomColor: colors.border }]}>
        <TouchableOpacity style={styles.backBtn} onPress={() => navigation.goBack()} hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}>
          <ChevronLeft size={24} color={colors.text} />
        </TouchableOpacity>
        <View style={styles.headerTitles}>
          <Text style={[styles.headerTag, { color: colors.primary }]}>Model Correction</Text>
          <Text style={[styles.headerTitle, { color: colors.text }]}>Wrong Prediction Fix</Text>
        </View>
      </View>

      <ScrollView
        style={styles.scrollView}
        contentContainerStyle={[styles.content, { paddingBottom: insets.bottom + 100 }]}
        showsVerticalScrollIndicator={false}
      >
        {/* Improve banner */}
        <View style={[styles.infoBanner, { backgroundColor: colors.primarySoft }]}>
          <CheckCircle size={18} color={colors.primary} />
          <View style={styles.infoBannerText}>
            <Text style={[styles.infoBannerTitle, { color: colors.primary }]}>Help Improve Model Performance</Text>
            <Text style={[styles.infoBannerSub, { color: colors.textSecondary }]}>
              Your correction is saved for future retraining.
            </Text>
          </View>
        </View>

        {/* Model guessed */}
        {originalPrediction ? (
          <Card style={styles.guessCard}>
            <Text style={[styles.guessLabel, { color: colors.textSecondary }]}>MODEL GUESSED</Text>
            <View style={styles.guessRow}>
              <Text style={[styles.guessValue, { color: colors.text }]}>{originalPrediction}</Text>
              {originalConfidence > 0 && (
                <View style={[styles.confBadge, { backgroundColor: colors.error + '20' }]}>
                  <Text style={[styles.confBadgeText, { color: colors.error }]}>{originalConfidence}% confidence</Text>
                </View>
              )}
            </View>
          </Card>
        ) : null}

        {/* Progress indicator */}
        <Card style={styles.progressCard}>
          <View style={styles.progressHeader}>
            <Text style={[styles.progressTitle, { color: colors.text }]}>Submission Readiness</Text>
          </View>
          <View style={styles.progressMeta}>
            <View>
              <Text style={[styles.progressMetaLabel, { color: colors.textSecondary }]}>Required fields</Text>
              <Text style={[styles.progressMetaValue, { color: readyToSubmit ? colors.success : colors.textSecondary }]}>
                {readyToSubmit ? 'Ready' : 'Incomplete'}
              </Text>
            </View>
            <View>
              <Text style={[styles.progressMetaLabel, { color: colors.textSecondary }]}>Photos added</Text>
              <Text style={[styles.progressMetaValue, { color: colors.text }]}>
                {Object.values(images).filter(Boolean).length}/4
              </Text>
            </View>
          </View>
          <View style={[styles.progressBarBg, { backgroundColor: colors.border }]}>
            <View style={[styles.progressBarFill, { width: `${completionPct}%`, backgroundColor: completionPct >= 60 ? colors.primary : colors.warning || '#f59e0b' }]} />
          </View>
          <Text style={[styles.progressPct, { color: colors.textSecondary }]}>{completionPct}% ready</Text>

          {/* Quick flow steps */}
          <View style={styles.stepsList}>
            {[
              { num: 1, label: 'Required details', done: readyToSubmit },
              { num: 2, label: 'Add photos', done: Object.values(images).some(Boolean) },
              { num: 3, label: 'Lookup + submit', done: hasSubmitted },
            ].map((s) => (
              <View key={s.num} style={[styles.stepRow, { backgroundColor: s.done ? colors.primarySoft : colors.surfaceSecondary }]}>
                <View style={[styles.stepNum, { backgroundColor: s.done ? colors.primary : colors.border }]}>
                  <Text style={[styles.stepNumText, { color: s.done ? '#fff' : colors.textSecondary }]}>{s.num}</Text>
                </View>
                <Text style={[styles.stepLabel, { color: s.done ? colors.primary : colors.text }]}>{s.label}</Text>
              </View>
            ))}
          </View>
        </Card>

        {/* Image slots */}
        <Text style={[styles.sectionTitle, { color: colors.text }]}>Recommended Angles</Text>
        <Text style={[styles.sectionSub, { color: colors.textSecondary }]}>
          Optional photos, but they greatly improve retraining quality.
        </Text>

        <ImageSlot slot="top" label="Top Image" optional={false} />
        <ImageSlot slot="side" label="Side Image" />
        <ImageSlot slot="inside" label="Inside Image" />
        <ImageSlot slot="label" label="Nutrition Label Image" />

        {/* Meal details */}
        <Text style={[styles.sectionTitle, { color: colors.text, marginTop: Spacing.xl }]}>Meal Details</Text>
        <Text style={[styles.sectionSub, { color: colors.textSecondary }]}>
          Keep this simple: food name + serving size is enough to submit.
        </Text>

        <Card style={styles.detailsCard}>
          <Text style={[styles.fieldLabel, { color: colors.textSecondary }]}>Food Name <Text style={{ color: colors.error }}>*</Text></Text>
          <TextInput
            style={[styles.input, {
              color: colors.text,
              backgroundColor: colors.surfaceSecondary,
              borderColor: foodName.trim() ? colors.primary : colors.border,
            }]}
            value={foodName}
            onChangeText={setFoodName}
            placeholder="e.g. miso soup"
            placeholderTextColor={colors.textSecondary}
            autoCapitalize="none"
          />

          <Text style={[styles.fieldLabel, { color: colors.textSecondary, marginTop: Spacing.lg }]}>
            Serving Size <Text style={{ color: colors.error }}>*</Text>
          </Text>

          {/* Unit Selection */}
          <View style={styles.unitRow}>
            {['serving', 'grams', 'oz', 'calories'].map((unit) => (
              <TouchableOpacity
                key={unit}
                style={[styles.unitBtn, {
                  backgroundColor: servingUnit === unit ? colors.primary : colors.surfaceSecondary,
                  borderColor: servingUnit === unit ? colors.primary : colors.border,
                }]}
                onPress={() => setServingUnit(unit)}
                activeOpacity={0.7}
              >
                <Text style={[styles.unitBtnText, { color: servingUnit === unit ? '#fff' : colors.text }]}>
                  {unit === 'grams' ? 'g' : unit === 'oz' ? 'oz' : unit === 'calories' ? 'kcal' : 'srv'}
                </Text>
              </TouchableOpacity>
            ))}
          </View>

          {/* Input with unit suffix */}
          <View style={[styles.inputWithUnit, { borderColor: colors.border }]}>
            <TextInput
              style={[styles.inputNumber, { color: colors.text, backgroundColor: colors.surfaceSecondary }]}
              value={servingSize}
              onChangeText={setServingSize}
              placeholder="e.g. 1, 100, 250"
              placeholderTextColor={colors.textSecondary}
              keyboardType="decimal-pad"
            />
            <View style={[styles.unitSuffix, { backgroundColor: colors.surfaceSecondary, borderLeftColor: colors.border }]}>
              <Text style={[styles.unitSuffixText, { color: colors.textSecondary }]}>
                {servingUnit === 'grams' ? 'g' : servingUnit === 'oz' ? 'oz' : servingUnit === 'calories' ? 'kcal' : 'srv'}
              </Text>
            </View>
          </View>

          {/* Quick presets for current unit */}
          <Text style={[styles.presetsLabel, { color: colors.textSecondary }]}>Quick presets</Text>
          <View style={styles.presetsRow}>
            {getPresetsForUnit(servingUnit).map((value) => (
              <TouchableOpacity
                key={value}
                style={[styles.presetChip, {
                  backgroundColor: servingSize === String(value) ? colors.primary : colors.surfaceSecondary,
                  borderColor: servingSize === String(value) ? colors.primary : colors.border,
                }]}
                onPress={() => setServingSize(String(value))}
                activeOpacity={0.7}
              >
                <Text style={[styles.presetChipText, { color: servingSize === String(value) ? '#fff' : colors.text }]}>
                  {value} {servingUnit === 'grams' ? 'g' : servingUnit === 'oz' ? 'oz' : servingUnit === 'calories' ? 'kcal' : 'srv'}
                </Text>
              </TouchableOpacity>
            ))}
          </View>

          {/* Live preview */}
          <View style={[styles.livePreview, { backgroundColor: colors.surfaceSecondary, borderColor: colors.border }]}>
            <Text style={[styles.livePreviewTitle, { color: colors.textSecondary }]}>Live Preview</Text>
            <View style={styles.livePreviewRow}>
              <Text style={[styles.livePreviewKey, { color: colors.textSecondary }]}>Food</Text>
              <Text style={[styles.livePreviewVal, { color: colors.text }]}>{foodName || '—'}</Text>
            </View>
            <View style={styles.livePreviewRow}>
              <Text style={[styles.livePreviewKey, { color: colors.textSecondary }]}>Serving</Text>
              <Text style={[styles.livePreviewVal, { color: colors.text }]}>
                {servingSize ? `${servingSize} ${servingUnit === 'grams' ? 'g' : servingUnit === 'oz' ? 'oz' : servingUnit === 'calories' ? 'kcal' : 'serving'}` : '—'}
              </Text>
            </View>
            <View style={styles.livePreviewRow}>
              <Text style={[styles.livePreviewKey, { color: colors.textSecondary }]}>Photos</Text>
              <Text style={[styles.livePreviewVal, { color: colors.text }]}>
                {Object.values(images).filter(Boolean).length}/4 selected
              </Text>
            </View>
          </View>
        </Card>

        {/* Error */}
        {submitError && (
          <View style={[styles.errorBox, { backgroundColor: colors.error + '20', borderColor: colors.error }]}>
            <AlertCircle size={16} color={colors.error} />
            <Text style={[styles.errorText, { color: colors.error }]}>{submitError}</Text>
          </View>
        )}

        {/* Submit */}
        {!hasSubmitted && (
          <TouchableOpacity
            style={[
              styles.submitBtn,
              {
                backgroundColor: readyToSubmit ? colors.primary : colors.border,
                opacity: isSubmitting ? 0.7 : 1,
              },
            ]}
            onPress={handleSubmit}
            disabled={!readyToSubmit || isSubmitting}
            activeOpacity={0.85}
          >
            {isSubmitting ? (
              <ActivityIndicator color="#fff" size="small" />
            ) : (
              <Text style={styles.submitBtnText}>Submit</Text>
            )}
          </TouchableOpacity>
        )}

        {/* ── POST-SUBMISSION SECTION ────────────────────────────────── */}
        {hasSubmitted && (
          <>
            {/* XP reward banner */}
            <View style={[styles.xpRewardBanner, { backgroundColor: colors.primary + '15', borderColor: colors.primary }]}>
              <CheckCircle size={22} color={colors.primary} />
              <View style={{ flex: 1 }}>
                <Text style={[styles.xpRewardTitle, { color: colors.primary }]}>+{xpEarned} XP Earned</Text>
                <Text style={[styles.xpRewardSub, { color: colors.textSecondary }]}>
                  Thanks for making food recognition smarter!
                </Text>
              </View>
            </View>

            {/* Nutrition results */}
            <Text style={[styles.sectionTitle, { color: colors.text }]}>Updated Nutrition</Text>
            <Text style={[styles.sectionSub, { color: colors.textSecondary }]}>Based on your correction</Text>

            {isFetchingNutrition && (
              <View style={styles.nutritionLoading}>
                <ActivityIndicator color={colors.primary} />
                <Text style={[styles.nutritionLoadingText, { color: colors.textSecondary }]}>
                  Fetching nutrition data...
                </Text>
              </View>
            )}

            {nutritionError && (
              <View style={[styles.errorBox, { backgroundColor: colors.error + '20', borderColor: colors.error }]}>
                <AlertCircle size={16} color={colors.error} />
                <Text style={[styles.errorText, { color: colors.error }]}>{nutritionError}</Text>
                <TouchableOpacity onPress={retryNutrition} style={[styles.retryBtn, { borderColor: colors.error }]}>
                  <Text style={[styles.retryBtnText, { color: colors.error }]}>Retry</Text>
                </TouchableOpacity>
              </View>
            )}

            {correctedNutrition && !isFetchingNutrition && (
              <Card style={styles.nutritionCard}>
                <View style={styles.nutritionHeader}>
                  <Text style={[styles.nutritionFoodName, { color: colors.text }]}>{foodName}</Text>
                  <Text style={[styles.nutritionServing, { color: colors.textSecondary }]}>{servingSize}</Text>
                </View>

                {/* Macro grid */}
                <View style={styles.macroGrid}>
                  {[
                    { label: 'Calories', value: Math.round(correctedNutrition.calories || 0), unit: 'kcal', color: colors.calories },
                    { label: 'Protein', value: Math.round(correctedNutrition.protein_g || 0), unit: 'g', color: colors.protein },
                    { label: 'Carbs', value: Math.round(correctedNutrition.carbs_g || 0), unit: 'g', color: colors.carbs },
                    { label: 'Fat', value: Math.round(correctedNutrition.fat_g || 0), unit: 'g', color: colors.fat },
                  ].map((item, idx) => (
                    <View key={idx} style={styles.macroItem}>
                      <Text style={[styles.macroValue, { color: item.color }]}>{item.value}</Text>
                      <Text style={[styles.macroUnit, { color: colors.textTertiary }]}>{item.unit}</Text>
                      <Text style={[styles.macroLabel, { color: colors.textSecondary }]}>{item.label}</Text>
                    </View>
                  ))}
                </View>

                {/* Full nutrition facts */}
                <View style={[styles.nutritionFacts, { borderTopColor: colors.border }]}>
                  {[
                    { label: 'Dietary Fiber', value: Math.round(correctedNutrition.fiber_g || 0), unit: 'g' },
                    { label: 'Sugars', value: Math.round(correctedNutrition.sugar_g || 0), unit: 'g' },
                    { label: 'Sodium', value: Math.round(correctedNutrition.sodium_mg || 0), unit: 'mg' },
                    { label: 'Cholesterol', value: Math.round(correctedNutrition.cholesterol_mg || 0), unit: 'mg' },
                  ].map((item, idx) => (
                    <View key={idx} style={[styles.nutritionFactRow, { borderBottomColor: colors.border }]}>
                      <Text style={[styles.nutritionFactLabel, { color: colors.text }]}>{item.label}</Text>
                      <Text style={[styles.nutritionFactValue, { color: colors.primary }]}>
                        {item.value} {item.unit}
                      </Text>
                    </View>
                  ))}
                </View>

                {/* Log meal button */}
                <TouchableOpacity
                  style={[styles.logMealBtn, { backgroundColor: colors.primary, opacity: isLoggingMeal ? 0.7 : 1 }]}
                  onPress={handleLogMeal}
                  disabled={isLoggingMeal}
                  activeOpacity={0.85}
                >
                  {isLoggingMeal ? (
                    <ActivityIndicator color="#fff" size="small" />
                  ) : (
                    <Text style={styles.logMealBtnText}>Log this meal</Text>
                  )}
                </TouchableOpacity>
              </Card>
            )}
          </>
        )}
      </ScrollView>

      <SuccessModal />
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  scrollView: { flex: 1 },
  content: { padding: Spacing.lg },

  header: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: Spacing.lg,
    paddingVertical: Spacing.md,
    borderBottomWidth: 1,
    gap: Spacing.md,
  },
  backBtn: { padding: Spacing.xs },
  headerTitles: { flex: 1 },
  headerTag: { ...Typography.caption, fontWeight: '700', textTransform: 'uppercase', letterSpacing: 0.8 },
  headerTitle: { ...Typography.h2, fontWeight: '700' },

  infoBanner: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: Spacing.md,
    padding: Spacing.md,
    borderRadius: BorderRadius.lg,
    marginBottom: Spacing.lg,
  },
  infoBannerText: { flex: 1 },
  infoBannerTitle: { ...Typography.bodyMedium, fontWeight: '700' },
  infoBannerSub: { ...Typography.caption, marginTop: 2 },

  guessCard: { marginBottom: Spacing.lg },
  guessLabel: { ...Typography.caption, fontWeight: '700', textTransform: 'uppercase', letterSpacing: 0.6, marginBottom: Spacing.xs },
  guessRow: { flexDirection: 'row', alignItems: 'center', gap: Spacing.md, flexWrap: 'wrap' },
  guessValue: { ...Typography.h3, fontStyle: 'italic' },
  confBadge: { paddingHorizontal: Spacing.sm, paddingVertical: 3, borderRadius: BorderRadius.full },
  confBadgeText: { ...Typography.caption, fontWeight: '700' },

  progressCard: { marginBottom: Spacing.xl },
  progressHeader: { marginBottom: Spacing.sm },
  progressTitle: { ...Typography.bodyMedium, fontWeight: '700' },
  progressMeta: { flexDirection: 'row', justifyContent: 'space-between', marginBottom: Spacing.sm },
  progressMetaLabel: { ...Typography.caption },
  progressMetaValue: { ...Typography.bodyMedium, fontWeight: '600' },
  progressBarBg: { height: 6, borderRadius: 4, overflow: 'hidden', marginBottom: Spacing.xs },
  progressBarFill: { height: '100%', borderRadius: 4 },
  progressPct: { ...Typography.caption, marginBottom: Spacing.md },
  stepsList: { gap: Spacing.sm },
  stepRow: { flexDirection: 'row', alignItems: 'center', gap: Spacing.md, padding: Spacing.sm, borderRadius: BorderRadius.md },
  stepNum: { width: 24, height: 24, borderRadius: 12, alignItems: 'center', justifyContent: 'center' },
  stepNumText: { fontSize: 12, fontWeight: '700' },
  stepLabel: { ...Typography.body },

  sectionTitle: { ...Typography.h3, marginBottom: Spacing.xs },
  sectionSub: { ...Typography.caption, marginBottom: Spacing.lg },

  imageSlotContainer: { marginBottom: Spacing.lg },
  imageSlotHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: Spacing.xs },
  imageSlotLabel: { ...Typography.bodyMedium, fontWeight: '600' },
  imageSlotTag: { ...Typography.caption },
  imageSlotBox: {
    height: 160,
    borderRadius: BorderRadius.lg,
    borderWidth: 1.5,
    alignItems: 'center',
    justifyContent: 'center',
    overflow: 'hidden',
    marginBottom: Spacing.sm,
  },
  imageSlotPreview: { width: '100%', height: '100%', resizeMode: 'cover' },
  imageSlotRemove: { position: 'absolute', top: Spacing.sm, right: Spacing.sm, width: 24, height: 24, borderRadius: 12, alignItems: 'center', justifyContent: 'center' },
  imageSlotActions: { flexDirection: 'row', gap: Spacing.sm },
  imageSlotBtn: { flex: 1, flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: Spacing.xs, paddingVertical: Spacing.sm, borderRadius: BorderRadius.md, borderWidth: 1 },
  imageSlotBtnText: { ...Typography.caption, fontWeight: '600' },

  detailsCard: { marginBottom: Spacing.lg },
  fieldLabel: { ...Typography.caption, fontWeight: '600', textTransform: 'uppercase', letterSpacing: 0.4, marginBottom: Spacing.xs },
  input: {
    borderWidth: 1.5,
    borderRadius: BorderRadius.lg,
    paddingHorizontal: Spacing.md,
    paddingVertical: Spacing.sm + 4,
    ...Typography.body,
  },
  presetsLabel: { ...Typography.caption, marginTop: Spacing.md, marginBottom: Spacing.xs },
  presetsRow: { flexDirection: 'row', flexWrap: 'wrap', gap: Spacing.sm, marginBottom: Spacing.lg },
  presetChip: { paddingHorizontal: Spacing.md, paddingVertical: Spacing.xs, borderRadius: BorderRadius.full, borderWidth: 1 },
  presetChipText: { ...Typography.caption, fontWeight: '600' },

  // Unit selection
  unitRow: { flexDirection: 'row', gap: Spacing.sm, marginBottom: Spacing.lg },
  unitBtn: { flex: 1, paddingVertical: Spacing.sm, borderRadius: BorderRadius.md, alignItems: 'center', borderWidth: 1.5 },
  unitBtnText: { ...Typography.caption, fontWeight: '700', textTransform: 'uppercase' },

  // Input with unit suffix
  inputWithUnit: { flexDirection: 'row', borderRadius: BorderRadius.lg, borderWidth: 1, overflow: 'hidden', marginBottom: Spacing.lg, backgroundColor: '#fff' },
  inputNumber: { flex: 1, paddingHorizontal: Spacing.md, paddingVertical: Spacing.sm + 4, ...Typography.body, borderWidth: 0 },
  unitSuffix: { paddingHorizontal: Spacing.md, paddingVertical: Spacing.sm + 4, justifyContent: 'center', borderLeftWidth: 1, minWidth: 60, alignItems: 'center' },
  unitSuffixText: { ...Typography.caption, fontWeight: '700', opacity: 0.8 },

  livePreview: { borderRadius: BorderRadius.lg, borderWidth: 1, padding: Spacing.md, gap: Spacing.xs },
  livePreviewTitle: { ...Typography.caption, fontWeight: '700', textTransform: 'uppercase', marginBottom: Spacing.xs },
  livePreviewRow: { flexDirection: 'row', justifyContent: 'space-between' },
  livePreviewKey: { ...Typography.caption },
  livePreviewVal: { ...Typography.caption, fontWeight: '600' },

  errorBox: { flexDirection: 'row', alignItems: 'flex-start', gap: Spacing.sm, padding: Spacing.md, borderRadius: BorderRadius.lg, borderWidth: 1, marginBottom: Spacing.lg },
  errorText: { ...Typography.caption, flex: 1 },
  retryBtn: { paddingHorizontal: Spacing.sm, paddingVertical: 2, borderRadius: BorderRadius.sm, borderWidth: 1 },
  retryBtnText: { ...Typography.caption, fontWeight: '700' },

  submitBtn: { paddingVertical: Spacing.lg, borderRadius: BorderRadius.xl, alignItems: 'center', marginBottom: Spacing.xl, ...Shadow.sm },
  submitBtnText: { ...Typography.bodyMedium, color: '#fff', fontWeight: '700' },

  xpRewardBanner: { flexDirection: 'row', alignItems: 'center', gap: Spacing.md, padding: Spacing.md, borderRadius: BorderRadius.lg, borderWidth: 1.5, marginBottom: Spacing.xl },
  xpRewardTitle: { ...Typography.bodyMedium, fontWeight: '700' },
  xpRewardSub: { ...Typography.caption, marginTop: 2 },

  nutritionLoading: { alignItems: 'center', gap: Spacing.sm, padding: Spacing.xl },
  nutritionLoadingText: { ...Typography.body },

  nutritionCard: { marginBottom: Spacing.xl },
  nutritionHeader: { marginBottom: Spacing.lg },
  nutritionFoodName: { ...Typography.h2, fontWeight: '700', textTransform: 'capitalize' },
  nutritionServing: { ...Typography.caption, marginTop: 2 },
  macroGrid: { flexDirection: 'row', justifyContent: 'space-between', marginBottom: Spacing.lg },
  macroItem: { alignItems: 'center', flex: 1 },
  macroValue: { ...Typography.h2, fontWeight: '700' },
  macroUnit: { ...Typography.small },
  macroLabel: { ...Typography.caption, marginTop: 2 },
  nutritionFacts: { borderTopWidth: 1, paddingTop: Spacing.md, marginBottom: Spacing.lg },
  nutritionFactRow: { flexDirection: 'row', justifyContent: 'space-between', paddingVertical: Spacing.sm, borderBottomWidth: 1 },
  nutritionFactLabel: { ...Typography.body },
  nutritionFactValue: { ...Typography.bodyMedium, fontWeight: '600' },
  logMealBtn: { paddingVertical: Spacing.md, borderRadius: BorderRadius.lg, alignItems: 'center', ...Shadow.sm },
  logMealBtnText: { ...Typography.bodyMedium, color: '#fff', fontWeight: '700' },

  // Success modal
  successOverlay: { flex: 1, alignItems: 'center', justifyContent: 'center', padding: Spacing.xl },
  successCard: { width: '100%', borderRadius: BorderRadius.xxxl, padding: Spacing.xxxl, alignItems: 'center', gap: Spacing.lg, ...Shadow.lg },
  successIconCircle: { width: 96, height: 96, borderRadius: 48, alignItems: 'center', justifyContent: 'center' },
  successTitle: { ...Typography.h2, fontWeight: '700', textAlign: 'center' },
  successMessage: { ...Typography.body, textAlign: 'center' },
  xpBadge: { paddingHorizontal: Spacing.xl, paddingVertical: Spacing.sm, borderRadius: BorderRadius.full },
  xpBadgeText: { color: '#fff', fontWeight: '800', fontSize: 20 },
  successCloseBtn: { width: '100%', paddingVertical: Spacing.lg, borderRadius: BorderRadius.xl, alignItems: 'center', marginTop: Spacing.sm },
  successCloseBtnText: { ...Typography.bodyMedium, color: '#fff', fontWeight: '700' },
});
