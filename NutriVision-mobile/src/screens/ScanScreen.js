import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  TextInput,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Image,
  Alert,
  ActivityIndicator,
  Modal,
  Pressable,
  Dimensions,
} from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useNavigation } from '@react-navigation/native';
import { useTheme } from '../theme/ThemeContext';
import { useAuth } from '../context/AuthContext';
import { useMeals } from '../context/MealContext';
import { Typography, Spacing, BorderRadius, Shadow } from '../theme';
import { Card, Button } from '../components';
import { Camera, Upload, Plus, Minus, X, Info } from 'lucide-react-native';
import { apiClient, apiRequest } from '../api/client';

// Portion presets matching backend
const PORTION_PRESETS = {
  small: { label: 'Small', multiplier: 0.68, grams: 115, ounces: 4 },
  medium: { label: 'Medium', multiplier: 1.0, grams: 170, ounces: 6 },
  large: { label: 'Large', multiplier: 1.32, grams: 225, ounces: 8 },
  extra_large: { label: 'Extra Large', multiplier: 1.68, grams: 285, ounces: 10 },
};

export default function ScanScreen() {
  const { colors } = useTheme();
  const insets = useSafeAreaInsets();
  const navigation = useNavigation();
  const { token } = useAuth();
  const { addMeal } = useMeals();

  // Step states
  const [step, setStep] = useState('upload'); // 'upload' | 'portions' | 'review'
  const [isUploading, setIsUploading] = useState(false);
  const [isFetchingNutrition, setIsFetchingNutrition] = useState(false);
  const [isSaving, setIsSaving] = useState(false);

  // Image & predictions
  const [imageUri, setImageUri] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [selectedPrediction, setSelectedPrediction] = useState(0);
  const [nutrition, setNutrition] = useState(null);
  const [scaledNutrition, setScaledNutrition] = useState(null);

  // Portion adjustments
  const [selectedPortionId, setSelectedPortionId] = useState('medium');
  const [customMultiplier, setCustomMultiplier] = useState(1.0);
  const [showFullNutrition, setShowFullNutrition] = useState(false);
  const [customGramsInput, setCustomGramsInput] = useState('');
  const [customUnit, setCustomUnit] = useState('g'); // 'g' or 'oz'

  // Error handling
  const [error, setError] = useState(null);

  // Scale nutrition based on portion
  useEffect(() => {
    if (!nutrition) return;
    const portion = PORTION_PRESETS[selectedPortionId];
    const multiplier = customMultiplier || portion.multiplier;
    const scaled = scaleNutrition(nutrition, multiplier);
    setScaledNutrition(scaled);
  }, [nutrition, selectedPortionId, customMultiplier]);

  const scaleNutrition = (nut, multiplier) => {
    const mult = Math.max(0.25, Math.min(multiplier, 4.0));
    return {
      calories: Math.round((nut.calories || 0) * mult * 10) / 10,
      protein_g: Math.round((nut.protein_g || 0) * mult * 10) / 10,
      carbs_g: Math.round((nut.carbs_g || 0) * mult * 10) / 10,
      fat_g: Math.round((nut.fat_g || 0) * mult * 10) / 10,
      fiber_g: Math.round((nut.fiber_g || 0) * mult * 10) / 10,
      sugar_g: Math.round((nut.sugar_g || 0) * mult * 10) / 10,
      sodium_mg: Math.round((nut.sodium_mg || 0) * mult * 10) / 10,
      cholesterol_mg: Math.round((nut.cholesterol_mg || 0) * mult * 10) / 10,
    };
  };

  const processImage = async (uri) => {
    setError(null);
    setImageUri(uri);
    setStep('portions');
    setIsUploading(true);
    setPredictions([]);
    setNutrition(null);

    try {
      const result = await apiClient.uploadImage('/predict', uri);
      if (result.predictions && result.predictions.length > 0) {
        const enhancedPredictions = result.predictions.map((p) => ({
          label: p.class.replace(/_/g, ' '),
          rawClass: p.class,
          confidence: Math.round(p.confidence),
          emoji: '🍽️',
        }));
        setPredictions(enhancedPredictions);
        setSelectedPrediction(0);
        fetchNutrition(enhancedPredictions[0].rawClass);
      } else {
        setError('Could not identify food in this image. Try another photo.');
        setStep('upload');
      }
    } catch (err) {
      console.warn('API /predict failed', err);
      setError('Could not process image. Please check that your backend is running.');
      setStep('upload');
    } finally {
      setIsUploading(false);
    }
  };

  const fetchNutrition = async (className) => {
    setIsFetchingNutrition(true);
    try {
      const cleanClassName = className.replace(/_/g, ' ');
      const data = await apiClient.post('/map/nutrition', { query: cleanClassName });
      if (data.nutrition) {
        setNutrition(data.nutrition);
      } else {
        setError('No nutrition data found for this food. Please select another option or adjust the meal name.');
        setNutrition(null);
      }
    } catch (error) {
      console.warn('Failed to fetch nutrition:', error);
      setError('Could not fetch nutrition data. Check your internet connection.');
      setNutrition(null);
    } finally {
      setIsFetchingNutrition(false);
    }
  };

  const handleSelectPrediction = (index) => {
    if (index === selectedPrediction) return;
    setSelectedPrediction(index);
    setError(null);
    const pred = predictions[index];
    if (pred) {
      fetchNutrition(pred.rawClass);
    }
  };

  const handleTakePhoto = async () => {
    try {
      const { status } = await ImagePicker.requestCameraPermissionsAsync();
      if (status !== 'granted') {
        Alert.alert('Permission Needed', 'Camera permissions are required to take photos.');
        return;
      }
      const result = await ImagePicker.launchCameraAsync({ allowsEditing: true, quality: 0.15 });
      if (!result.canceled) {
        processImage(result.assets[0].uri);
      }
    } catch (error) {
      Alert.alert('Camera Error', 'Could not access camera. Try uploading a photo instead.');
    }
  };

  const handleUploadPhoto = async () => {
    try {
      const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
      if (status !== 'granted') {
        Alert.alert('Permission Needed', 'Gallery permissions are required to upload photos.');
        return;
      }
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: [ImagePicker.MediaType.image],
        allowsEditing: true,
        quality: 0.15,
      });
      if (!result.canceled) {
        processImage(result.assets[0].uri);
      }
    } catch (error) {
      Alert.alert('Gallery Error', 'Could not access photo gallery.');
    }
  };

  const handleConfirmPortion = () => {
    if (!nutrition || !scaledNutrition) {
      setError('Please wait for nutrition data to load.');
      return;
    }
    setStep('review');
  };

  const handleSaveMeal = async () => {
    if (!predictions[selectedPrediction] || !scaledNutrition) {
      setError('Please select a food and wait for nutrition data.');
      return;
    }

    if (!token) {
      setError('Not authenticated. Please sign in again to save meals.');
      return;
    }

    setIsSaving(true);
    setError(null);

    try {
      const portion = PORTION_PRESETS[selectedPortionId];
      const multiplier = customMultiplier || portion.multiplier;
      const portionMultiplier = Math.max(0.25, Math.min(multiplier, 4.0));
      const foodClass = predictions[selectedPrediction].rawClass;
      const foodLabel = predictions[selectedPrediction].label;

      let sourceNutrition = nutrition;
      try {
        const latestNutrition = await apiClient.post('/map/nutrition', {
          query: foodClass.replace(/_/g, ' '),
        });
        if (latestNutrition?.nutrition) {
          sourceNutrition = latestNutrition.nutrition;
        }
      } catch {
        // Keep the current estimate when live refetch fails.
      }

      const effectiveScaledNutrition = scaleNutrition(sourceNutrition || nutrition || {}, portionMultiplier);

      const servingDisplay = customGramsInput
        ? `${customGramsInput} ${customUnit}`
        : `${Math.round((portion?.grams || 170) * portionMultiplier)} g`;

      const response = await apiRequest('/map/log', {
        method: 'POST',
        token,
        body: {
          food_label: foodClass,
          display_name: foodLabel,
          portion_id: selectedPortionId,
          portion_multiplier: portionMultiplier,
          serving_size: servingDisplay,
          nutrition: effectiveScaledNutrition,
          raw_nutrition: sourceNutrition || {},
          prediction: predictions[selectedPrediction],
          source: 'mobile_camera',
          image_url: imageUri,
        },
      });

      if (response.status === 'saved') {
        // Update shared meal context from backend-normalized entry to keep all screens in sync.
        const saved = response.entry || {};
        const nut = saved.nutrition || effectiveScaledNutrition || {};
        addMeal(
          {
            id: saved.timestamp,
            timestamp: saved.timestamp,
            name: foodLabel,
            calories: Math.round(Number(saved.calories ?? nut.calories) || 0),
            protein: Math.round(Number(saved.protein ?? nut.protein_g ?? nut.protein) || 0),
            carbs: Math.round(Number(saved.carbs ?? nut.carbs_g ?? nut.carbs) || 0),
            fat: Math.round(Number(saved.fat ?? nut.fat_g ?? nut.fat) || 0),
            fiber: Number(saved.fiber ?? nut.fiber_g ?? nut.fiber ?? 0),
            sugar: Number(saved.sugar ?? nut.sugar_g ?? nut.sugar ?? 0),
            water: Number(saved.water ?? nut.water ?? 0),
            choline: Number(saved.choline ?? nut.choline_mg ?? nut.choline ?? 0),
            vitamins: saved.vitamins || {},
            minerals: saved.minerals || {},
            aminoAcids: saved.aminoAcids || saved.amino_acids || nut.aminoAcids || nut.amino_acids || {},
            fattyAcids: saved.fattyAcids || saved.fatty_acids || nut.fattyAcids || nut.fatty_acids || {},
            nutrients: nut,
            source: 'mobile_camera',
            portionMultiplier,
            servingSize: servingDisplay,
          },
          imageUri,
        );
        Alert.alert('Success', 'Meal saved to your log!', [
          {
            text: 'View Log',
            onPress: () => {
              handleReset();
              navigation.navigate('Home');
            },
          },
          {
            text: 'Done',
            onPress: () => handleReset(),
          },
        ]);
      }
    } catch (err) {
      if (Number(err?.status || 0) === 401) {
        setError('Session expired. Please sign in again, then retry saving this meal.');
      } else {
        setError('Could not save meal. Please check your internet connection and try again.');
      }
    } finally {
      setIsSaving(false);
    }
  };

  const handleReset = () => {
    setStep('upload');
    setImageUri(null);
    setPredictions([]);
    setNutrition(null);
    setScaledNutrition(null);
    setSelectedPrediction(0);
    setSelectedPortionId('medium');
    setCustomMultiplier(1.0);
    setCustomGramsInput('');
    setCustomUnit('g');
    setError(null);
  };

  const handleChangePortion = (delta) => {
    setCustomMultiplier((prev) => {
      const next = Math.max(0.25, Math.min(prev + delta * 0.25, 4.0));
      return Math.round(next * 100) / 100;
    });
  };


  return (
    <View style={[styles.container, { backgroundColor: colors.background, paddingTop: insets.top }]}>
      <ScrollView
        style={styles.scrollView}
        contentContainerStyle={[styles.content, { paddingBottom: insets.bottom + 80 }]}
        showsVerticalScrollIndicator={false}
      >
        {/* Upload Step */}
        {step === 'upload' && (
          <>
            <Card style={styles.uploadCard}>
              <View style={[styles.uploadArea, { borderColor: colors.border, backgroundColor: colors.surfaceSecondary }]}>
                <View style={[styles.uploadIconCircle, { backgroundColor: colors.primarySoft }]}>
                  <Camera size={40} color={colors.primary} />
                </View>
                <Text style={[styles.uploadTitle, { color: colors.text }]}>Scan Your Meal</Text>
                <Text style={[styles.uploadSubtitle, { color: colors.textSecondary }]}>
                  Take a photo or upload an image to identify your food and get detailed nutrition info
                </Text>
              </View>
            </Card>

            <View style={styles.actionRow}>
              <TouchableOpacity
                style={[styles.actionBtn, { backgroundColor: colors.primary }]}
                onPress={handleTakePhoto}
                activeOpacity={0.85}
              >
                <Camera size={22} color="#fff" />
                <Text style={styles.actionBtnText}>Take Photo</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[styles.actionBtn, { backgroundColor: colors.surface, borderColor: colors.border, borderWidth: 1 }]}
                onPress={handleUploadPhoto}
                activeOpacity={0.85}
              >
                <Upload size={22} color={colors.primary} />
                <Text style={[styles.actionBtnText, { color: colors.primary }]}>Upload</Text>
              </TouchableOpacity>
            </View>

            <Card style={styles.tipsCard}>
              <Text style={[styles.tipsTitle, { color: colors.text }]}>📌 Tips for Best Results</Text>
              {[
                'Place your food on a clean, contrasting surface',
                'Ensure bright, even lighting (avoid shadows)',
                'Capture the entire plate in frame',
                'Take the photo from directly above',
              ].map((tip, idx) => (
                <View key={idx} style={styles.tipRow}>
                  <Text style={[styles.tipBullet, { color: colors.primary }]}>•</Text>
                  <Text style={[styles.tipText, { color: colors.textSecondary }]}>{tip}</Text>
                </View>
              ))}
            </Card>
          </>
        )}

        {/* Portions Step */}
        {step === 'portions' && (
          <>
            <Card style={styles.previewCard}>
              <View style={[styles.previewImage, { backgroundColor: colors.surfaceSecondary }]}>
                {imageUri && <Image source={{ uri: imageUri }} style={{ width: '100%', height: '100%', resizeMode: 'cover' }} />}
                {!isUploading && predictions.length > 0 && (
                  <View style={[styles.confidenceBadge, { backgroundColor: colors.primary }]}>
                    <Text style={styles.confidenceText}>{predictions[selectedPrediction].confidence}% match</Text>
                  </View>
                )}
              </View>
            </Card>

            {isUploading ? (
              <View style={styles.loadingContainer}>
                <ActivityIndicator size="large" color={colors.primary} />
                <Text style={[styles.loadingText, { color: colors.textSecondary, marginTop: 16 }]}>
                  Running AI vision analysis...
                </Text>
              </View>
            ) : (
              <>
                {error && (
                  <View style={[styles.errorBox, { backgroundColor: colors.error + '20', borderColor: colors.error }]}>
                    <Text style={[styles.errorText, { color: colors.error }]}>{error}</Text>
                  </View>
                )}

                <Text style={[styles.sectionTitle, { color: colors.text }]}>What did you eat?</Text>
                <Text style={[styles.sectionSubtitle, { color: colors.textSecondary }]}>
                  Select the prediction that matches your meal
                </Text>

                <View style={styles.predictionsCol}>
                  {predictions.map((pred, index) => (
                    <TouchableOpacity
                      key={index}
                      style={[
                        styles.predictionCard,
                        {
                          backgroundColor: colors.surface,
                          borderColor: index === selectedPrediction ? colors.primary : colors.border,
                          borderWidth: index === selectedPrediction ? 2 : 1,
                          ...Shadow.sm,
                        },
                      ]}
                      onPress={() => handleSelectPrediction(index)}
                      activeOpacity={0.7}
                    >
                      <View style={[styles.predEmoji, { backgroundColor: colors.primarySoft }]}>
                        <Text style={{ fontSize: 22 }}>{pred.emoji}</Text>
                      </View>
                      <View style={styles.predInfo}>
                        <Text style={[styles.predLabel, { color: colors.text }]}>{pred.label}</Text>
                        <Text style={[styles.predConf, { color: colors.textSecondary }]}>
                          {pred.confidence}% confidence
                        </Text>
                      </View>
                      {index === selectedPrediction && (
                        <View style={[styles.checkMark, { backgroundColor: colors.primary }]}>
                          <Text style={styles.checkText}>✓</Text>
                        </View>
                      )}
                    </TouchableOpacity>
                  ))}
                </View>

                {/* Wrong Prediction Button */}
                <TouchableOpacity
                  style={[styles.wrongPredictionBtn, { backgroundColor: colors.primary }]}
                  onPress={() => navigation.navigate('WrongPrediction', {
                    originalPrediction: predictions[selectedPrediction]?.label || '',
                    originalConfidence: predictions[selectedPrediction]?.confidence || 0,
                    imageUri: imageUri,
                  })}
                  activeOpacity={0.75}
                >
                  <Text style={[styles.wrongPredictionText, { color: '#fff' }]}>
                    ✕  Wrong Prediction? Tap to correct
                  </Text>
                </TouchableOpacity>

                <Text style={[styles.sectionTitle, { color: colors.text, marginTop: Spacing.xl }]}>
                  Adjust Serving Size
                </Text>

                <Card style={styles.portionCard}>
                  {isFetchingNutrition ? (
                    <View style={styles.loadingContainer}>
                      <ActivityIndicator color={colors.primary} size="small" />
                      <Text style={[styles.loadingText, { color: colors.textSecondary, fontSize: 12 }]}>
                        Fetching nutrition...
                      </Text>
                    </View>
                  ) : nutrition ? (
                    <>
                      <Text style={[styles.portionLabel, { color: colors.text, marginBottom: Spacing.md }]}>
                        Portion Size
                      </Text>

                      <View style={styles.portionButtonsRow}>
                        {Object.entries(PORTION_PRESETS).map(([id, preset]) => (
                          <TouchableOpacity
                            key={id}
                            style={[
                              styles.portionBtn,
                              {
                                backgroundColor: selectedPortionId === id ? colors.primary : colors.surfaceSecondary,
                                borderColor: selectedPortionId === id ? colors.primary : colors.textSecondary,
                                borderWidth: selectedPortionId === id ? 2 : 1,
                              },
                            ]}
                            onPress={() => {
                              setSelectedPortionId(id);
                              setCustomMultiplier(preset.multiplier);
                            }}
                            activeOpacity={0.7}
                          >
                            <Text
                              style={[
                                styles.portionBtnText,
                                { color: selectedPortionId === id ? '#fff' : colors.text },
                              ]}
                            >
                              {preset.label}
                            </Text>
                            <Text
                              style={[
                                styles.portionBtnSubtext,
                                { color: selectedPortionId === id ? '#f5f5f5' : colors.textSecondary, fontSize: 11 },
                              ]}
                            >
                              {preset.grams}g
                            </Text>
                          </TouchableOpacity>
                        ))}
                      </View>

                      {/* Custom gram/oz input */}
                      <View style={[styles.customGramsContainer, { borderColor: colors.border, backgroundColor: colors.surfaceSecondary }]}>
                        <Text style={[styles.customGramsTitle, { color: colors.text }]}>Custom Amount</Text>
                        <View style={styles.customGramsRow}>
                          <View style={[styles.customUnitToggle]}>
                            {['g', 'oz'].map((u) => (
                              <TouchableOpacity
                                key={u}
                                style={[styles.customUnitBtn, { backgroundColor: customUnit === u ? colors.primary : colors.surface, borderColor: colors.border }]}
                                onPress={() => {
                                  setCustomUnit(u);
                                  if (customGramsInput) {
                                    const grams = u === 'oz'
                                      ? parseFloat(customGramsInput) * 28.35
                                      : parseFloat(customGramsInput);
                                    if (!isNaN(grams) && grams > 0) setCustomMultiplier(Math.round((grams / 170) * 100) / 100);
                                  }
                                }}
                                activeOpacity={0.7}
                              >
                                <Text style={[styles.customUnitBtnText, { color: customUnit === u ? '#fff' : colors.text }]}>{u}</Text>
                              </TouchableOpacity>
                            ))}
                          </View>
                          <TextInput
                            style={[styles.customGramsInput, { color: colors.text, backgroundColor: colors.surface, borderColor: colors.border }]}
                            value={customGramsInput}
                            onChangeText={(val) => {
                              setCustomGramsInput(val);
                              const num = parseFloat(val);
                              if (!isNaN(num) && num > 0) {
                                const grams = customUnit === 'oz' ? num * 28.35 : num;
                                setCustomMultiplier(Math.round((grams / 170) * 100) / 100);
                                setSelectedPortionId('custom');
                              }
                            }}
                            placeholder={customUnit === 'g' ? 'e.g. 150' : 'e.g. 5.5'}
                            placeholderTextColor={colors.textSecondary}
                            keyboardType="decimal-pad"
                          />
                          <View style={[styles.customGramsSuffix, { backgroundColor: colors.surface, borderColor: colors.border }]}>
                            <Text style={[styles.customGramsSuffixText, { color: colors.textSecondary }]}>{customUnit}</Text>
                          </View>
                        </View>
                        <Text style={[styles.customGramsHint, { color: colors.textSecondary }]}>
                          = {Math.round(customMultiplier * 170)}{customUnit === 'oz' ? ` g (${(customMultiplier * 6).toFixed(1)} oz)` : 'g'} • {customMultiplier.toFixed(2)}x multiplier
                        </Text>
                      </View>

                      <Text style={[styles.sectionTitle, { color: colors.text, marginTop: Spacing.lg }]}>
                        Nutrition Estimate (Scaled)
                      </Text>

                      <View style={styles.macroGrid}>
                        {[
                          { label: 'Calories', value: scaledNutrition?.calories, unit: 'kcal', color: colors.calories },
                          { label: 'Protein', value: scaledNutrition?.protein_g, unit: 'g', color: colors.protein },
                          { label: 'Carbs', value: scaledNutrition?.carbs_g, unit: 'g', color: colors.carbs },
                          { label: 'Fat', value: scaledNutrition?.fat_g, unit: 'g', color: colors.fat },
                        ].map((item, idx) => (
                          <View key={idx} style={styles.macroItem}>
                            <Text style={[styles.macroValue, { color: item.color }]}>{item.value || 0}</Text>
                            <Text style={[styles.macroUnit, { color: colors.textTertiary }]}>{item.unit}</Text>
                            <Text style={[styles.macroLabel, { color: colors.textSecondary }]}>{item.label}</Text>
                          </View>
                        ))}
                      </View>

                      <TouchableOpacity
                        style={[styles.viewDetailsBtn, { borderColor: colors.primary }]}
                        onPress={() => setShowFullNutrition(true)}
                        activeOpacity={0.7}
                      >
                        <Info size={16} color={colors.primary} />
                        <Text style={[styles.viewDetailsText, { color: colors.primary }]}>View Full Nutrition Facts</Text>
                      </TouchableOpacity>
                    </>
                  ) : (
                    <Text style={[styles.errorText, { color: colors.error }]}>
                      Unable to load nutrition. Try selecting a different prediction.
                    </Text>
                  )}
                </Card>

                <View style={styles.portionActions}>
                  <Button
                    title="Back"
                    variant="outline"
                    size="lg"
                    onPress={handleReset}
                    style={{ flex: 1 }}
                  />
                  <Button
                    title="Confirm"
                    size="lg"
                    onPress={handleConfirmPortion}
                    disabled={!nutrition || isFetchingNutrition}
                    style={{ flex: 1 }}
                  />
                </View>
              </>
            )}
          </>
        )}

        {/* Review Step */}
        {step === 'review' && scaledNutrition && (
          <>
            <Card style={styles.reviewCard}>
              <View style={[styles.reviewImage, { backgroundColor: colors.surfaceSecondary }]}>
                {imageUri && <Image source={{ uri: imageUri }} style={{ width: '100%', height: '100%', resizeMode: 'cover' }} />}
              </View>
            </Card>

            <View style={styles.reviewInfo}>
              <Text style={[styles.reviewTitle, { color: colors.text }]}>{predictions[selectedPrediction]?.label}</Text>
              <Text style={[styles.reviewSubtitle, { color: colors.textSecondary }]}>
                {PORTION_PRESETS[selectedPortionId]?.label} • {PORTION_PRESETS[selectedPortionId]?.grams}g
              </Text>
            </View>

            <Card style={styles.reviewNutritionCard}>
              <View style={styles.reviewMacroGrid}>
                {[
                  { label: 'Calories', value: Math.round(scaledNutrition.calories), unit: 'kcal', color: colors.calories },
                  { label: 'Protein', value: Math.round(scaledNutrition.protein_g), unit: 'g', color: colors.protein },
                  { label: 'Carbs', value: Math.round(scaledNutrition.carbs_g), unit: 'g', color: colors.carbs },
                  { label: 'Fat', value: Math.round(scaledNutrition.fat_g), unit: 'g', color: colors.fat },
                ].map((item, idx) => (
                  <View key={idx} style={styles.reviewMacroItem}>
                    <Text style={[styles.reviewMacroValue, { color: item.color }]}>{item.value}</Text>
                    <Text style={[styles.reviewMacroUnit, { color: colors.textTertiary }]}>{item.unit}</Text>
                    <Text style={[styles.reviewMacroLabel, { color: colors.textSecondary }]}>{item.label}</Text>
                  </View>
                ))}
              </View>
            </Card>

            <TouchableOpacity
              style={[styles.viewDetailsBtn, { borderColor: colors.primary, marginBottom: Spacing.lg }]}
              onPress={() => setShowFullNutrition(true)}
              activeOpacity={0.7}
            >
              <Info size={16} color={colors.primary} />
              <Text style={[styles.viewDetailsText, { color: colors.primary }]}>View Full Nutrition Label</Text>
            </TouchableOpacity>

            {error && (
              <View style={[styles.errorBox, { backgroundColor: colors.error + '20', borderColor: colors.error }]}>
                <Text style={[styles.errorText, { color: colors.error }]}>{error}</Text>
              </View>
            )}

            {/* Wrong prediction link */}
            <TouchableOpacity
              style={styles.wrongPredictionLink}
              onPress={() =>
                navigation.navigate('WrongPrediction', {
                  originalPrediction: predictions[selectedPrediction]?.label || '',
                  originalConfidence: predictions[selectedPrediction]?.confidence || 0,
                  imageUri: imageUri,
                })
              }
              activeOpacity={0.7}
            >
              <Text style={[styles.wrongPredictionText, { color: colors.primary }]}>Wrong prediction?</Text>
              <Text style={[styles.wrongPredictionSub, { color: colors.textSecondary }]}>Help improve the model</Text>
            </TouchableOpacity>

            <View style={styles.reviewActions}>
              <Button
                title="Back"
                variant="outline"
                size="lg"
                onPress={() => setStep('portions')}
                style={{ flex: 1 }}
              />
              <Button
                title={isSaving ? 'Saving...' : 'Add to Log'}
                size="lg"
                onPress={handleSaveMeal}
                disabled={isSaving}
                style={{ flex: 1 }}
              />
            </View>
          </>
        )}
      </ScrollView>

      {/* Full Nutrition Facts Modal */}
      <Modal visible={showFullNutrition} transparent animationType="slide" onRequestClose={() => setShowFullNutrition(false)}>
        <View style={[styles.modalOverlay, { backgroundColor: '#00000060' }]}>
          <View style={[styles.nutritionSheet, { backgroundColor: colors.background }]}>
            <View style={styles.nutritionSheetHeader}>
              <Text style={[styles.nutritionSheetTitle, { color: colors.text }]}>Nutrition Facts</Text>
              <TouchableOpacity onPress={() => setShowFullNutrition(false)} hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}>
                <X size={24} color={colors.text} />
              </TouchableOpacity>
            </View>

            <ScrollView style={styles.nutritionSheetBody} showsVerticalScrollIndicator={false}>
              <Text style={[styles.nutritionSheetSubtitle, { color: colors.textSecondary, marginBottom: Spacing.lg }]}>
                {predictions[selectedPrediction]?.label} • {PORTION_PRESETS[selectedPortionId]?.label}
              </Text>

              {scaledNutrition && (
                <View style={styles.nutritionFactsList}>
                  {[
                    { label: 'Calories', value: Math.round(scaledNutrition.calories), unit: 'kcal' },
                    { label: 'Protein', value: Math.round(scaledNutrition.protein_g), unit: 'g' },
                    { label: 'Carbohydrates', value: Math.round(scaledNutrition.carbs_g), unit: 'g' },
                    { label: 'Dietary Fiber', value: Math.round(scaledNutrition.fiber_g), unit: 'g' },
                    { label: 'Sugars', value: Math.round(scaledNutrition.sugar_g), unit: 'g' },
                    { label: 'Total Fat', value: Math.round(scaledNutrition.fat_g), unit: 'g' },
                    { label: 'Sodium', value: Math.round(scaledNutrition.sodium_mg), unit: 'mg' },
                    { label: 'Cholesterol', value: Math.round(scaledNutrition.cholesterol_mg), unit: 'mg' },
                  ].map((item, idx) => (
                    <View key={idx} style={[styles.nutritionFactRow, { borderBottomColor: colors.border }]}>
                      <Text style={[styles.nutritionFactLabel, { color: colors.text }]}>{item.label}</Text>
                      <Text style={[styles.nutritionFactValue, { color: colors.primary }]}>
                        {item.value} {item.unit}
                      </Text>
                    </View>
                  ))}
                </View>
              )}

              <Text style={[styles.disclaimerText, { color: colors.textSecondary, marginTop: Spacing.xl }]}>
                Values are estimated using AI image analysis and nutrition APIs. Actual nutrition may vary based on real ingredients, preparation method, and portion size.
              </Text>
            </ScrollView>

            <TouchableOpacity
              style={[styles.closeBtn, { backgroundColor: colors.primary }]}
              onPress={() => setShowFullNutrition(false)}
              activeOpacity={0.85}
            >
              <Text style={styles.closeBtnText}>Close</Text>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>
    </View>
  );
}


const styles = StyleSheet.create({
  container: { flex: 1 },
  scrollView: { flex: 1 },
  content: { padding: Spacing.lg },

  // Upload Step
  uploadCard: { padding: 0, overflow: 'hidden', marginBottom: Spacing.lg },
  uploadArea: {
    alignItems: 'center',
    justifyContent: 'center',
    padding: Spacing.xxxl,
    borderWidth: 2,
    borderStyle: 'dashed',
    borderRadius: BorderRadius.lg,
    gap: Spacing.md,
  },
  uploadIconCircle: {
    width: 80,
    height: 80,
    borderRadius: 40,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: Spacing.sm,
  },
  uploadTitle: { ...Typography.h2, textAlign: 'center' },
  uploadSubtitle: { ...Typography.body, textAlign: 'center', maxWidth: 280 },

  actionRow: { flexDirection: 'row', gap: Spacing.md, marginBottom: Spacing.xl },
  actionBtn: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: Spacing.lg,
    borderRadius: BorderRadius.xl,
    gap: Spacing.sm,
    ...Shadow.sm,
  },
  actionBtnText: { ...Typography.bodyMedium, color: '#fff', fontWeight: '700' },

  tipsCard: { gap: Spacing.sm },
  tipsTitle: { ...Typography.bodyMedium, fontWeight: '700', marginBottom: Spacing.xs },
  tipRow: { flexDirection: 'row', gap: Spacing.sm, alignItems: 'flex-start' },
  tipBullet: { fontSize: 16, lineHeight: 20 },
  tipText: { ...Typography.caption, lineHeight: 20, flex: 1 },

  // Preview & Predictions
  previewCard: { padding: 0, overflow: 'hidden', marginBottom: Spacing.lg },
  previewImage: { height: 240, alignItems: 'center', justifyContent: 'center' },
  confidenceBadge: {
    position: 'absolute',
    top: Spacing.md,
    right: Spacing.md,
    paddingHorizontal: Spacing.md,
    paddingVertical: Spacing.xs,
    borderRadius: BorderRadius.full,
  },
  confidenceText: { color: '#fff', ...Typography.captionMedium, fontWeight: '700' },

  sectionTitle: { ...Typography.h3, marginBottom: Spacing.xs },
  sectionSubtitle: { ...Typography.caption, marginBottom: Spacing.lg },

  predictionsCol: { gap: Spacing.md, marginBottom: Spacing.md },
  predictionCard: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: Spacing.md,
    borderRadius: BorderRadius.lg,
    gap: Spacing.md,
    ...Shadow.sm,
  },
  predEmoji: { width: 48, height: 48, borderRadius: 24, alignItems: 'center', justifyContent: 'center' },
  predInfo: { flex: 1, gap: 2 },
  predLabel: { ...Typography.bodyMedium, textTransform: 'capitalize', fontWeight: '600' },
  predConf: { ...Typography.caption },
  checkMark: { width: 28, height: 28, borderRadius: 14, alignItems: 'center', justifyContent: 'center' },
  checkText: { color: '#fff', fontWeight: '700', fontSize: 14 },

  // Wrong Prediction
  wrongPredictionBtn: { paddingVertical: Spacing.md + 2, paddingHorizontal: Spacing.lg, borderRadius: BorderRadius.lg, alignItems: 'center', marginBottom: Spacing.lg },
  wrongPredictionText: { ...Typography.bodyMedium, fontWeight: '800', color: '#fff', letterSpacing: 0.3 },

  // Portion Selection
  portionCard: { marginBottom: Spacing.lg, padding: Spacing.lg },
  portionLabel: { ...Typography.bodyMedium, fontWeight: '600' },
  portionButtonsRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    gap: Spacing.sm,
    marginBottom: Spacing.lg,
    flexWrap: 'wrap',
  },
  portionBtn: {
    flex: 1,
    minWidth: Dimensions.get('window').width / 4 - Spacing.lg - 2,
    alignItems: 'center',
    paddingVertical: Spacing.md,
    paddingHorizontal: Spacing.sm,
    borderRadius: BorderRadius.lg,
    ...Shadow.xs,
  },
  portionBtnText: { ...Typography.bodyMedium, fontWeight: '600', textAlign: 'center' },
  portionBtnSubtext: { marginTop: 2, fontWeight: '500' },

  // Custom gram/oz input
  customGramsContainer: { borderRadius: BorderRadius.lg, borderWidth: 1, padding: Spacing.md, marginBottom: Spacing.lg },
  customGramsTitle: { ...Typography.caption, fontWeight: '700', textTransform: 'uppercase', marginBottom: Spacing.sm },
  customGramsRow: { flexDirection: 'row', alignItems: 'center', gap: Spacing.sm },
  customUnitToggle: { flexDirection: 'row', gap: 4 },
  customUnitBtn: { paddingVertical: Spacing.xs, paddingHorizontal: Spacing.sm, borderRadius: BorderRadius.md, borderWidth: 1 },
  customUnitBtnText: { ...Typography.caption, fontWeight: '700' },
  customGramsInput: { flex: 1, borderRadius: BorderRadius.md, borderWidth: 1, paddingHorizontal: Spacing.sm, paddingVertical: Spacing.xs, ...Typography.body, textAlign: 'center' },
  customGramsSuffix: { paddingHorizontal: Spacing.sm, paddingVertical: Spacing.xs, borderRadius: BorderRadius.md, borderWidth: 1 },
  customGramsSuffixText: { ...Typography.caption, fontWeight: '700' },
  customGramsHint: { ...Typography.small, marginTop: Spacing.xs },

  customMultiplierRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.md,
    marginBottom: Spacing.lg,
  },
  customLabel: { ...Typography.bodyMedium, fontWeight: '600', width: 60 },
  multiplierBtn: {
    width: 40,
    height: 40,
    borderRadius: BorderRadius.lg,
    alignItems: 'center',
    justifyContent: 'center',
    ...Shadow.xs,
  },
  multiplierDisplay: {
    flex: 1,
    paddingVertical: Spacing.sm,
    paddingHorizontal: Spacing.md,
    borderRadius: BorderRadius.lg,
    alignItems: 'center',
    ...Shadow.xs,
  },
  multiplierText: { ...Typography.bodyMedium, fontWeight: '700' },

  macroGrid: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: Spacing.lg,
  },
  macroItem: { alignItems: 'center', flex: 1, gap: 2 },
  macroValue: { ...Typography.h2, fontWeight: '700' },
  macroUnit: { ...Typography.small, textTransform: 'lowercase' },
  macroLabel: { ...Typography.caption, marginTop: 2 },

  viewDetailsBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: Spacing.sm,
    paddingVertical: Spacing.md,
    paddingHorizontal: Spacing.lg,
    borderWidth: 1,
    borderRadius: BorderRadius.lg,
  },
  viewDetailsText: { ...Typography.bodyMedium, fontWeight: '600' },

  portionActions: {
    flexDirection: 'row',
    gap: Spacing.md,
    marginTop: Spacing.xl,
  },

  // Review Step
  reviewCard: { padding: 0, overflow: 'hidden', marginBottom: Spacing.lg },
  reviewImage: { height: 200, alignItems: 'center', justifyContent: 'center' },

  reviewInfo: { marginBottom: Spacing.lg },
  reviewTitle: { ...Typography.h2, marginBottom: Spacing.xs },
  reviewSubtitle: { ...Typography.caption },

  reviewNutritionCard: { marginBottom: Spacing.lg, padding: Spacing.lg },
  reviewMacroGrid: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  reviewMacroItem: { alignItems: 'center', flex: 1 },
  reviewMacroValue: { ...Typography.h1, fontWeight: '700' },
  reviewMacroUnit: { ...Typography.small },
  reviewMacroLabel: { ...Typography.caption, marginTop: 2 },

  reviewActions: {
    flexDirection: 'row',
    gap: Spacing.md,
    marginTop: Spacing.xl,
  },

  wrongPredictionLink: {
    alignItems: 'center',
    paddingVertical: Spacing.md,
    marginBottom: Spacing.xs,
  },
  wrongPredictionText: { ...Typography.bodyMedium, fontWeight: '700', textDecorationLine: 'underline' },
  wrongPredictionSub: { ...Typography.caption, marginTop: 2 },

  // Error & Loading
  errorBox: {
    padding: Spacing.md,
    borderRadius: BorderRadius.lg,
    marginBottom: Spacing.lg,
    borderWidth: 1,
  },
  errorText: { ...Typography.body, fontWeight: '500' },

  loadingContainer: {
    padding: Spacing.xxxl,
    alignItems: 'center',
    justifyContent: 'center',
  },
  loadingText: { ...Typography.body },

  // Full Nutrition Modal
  modalOverlay: {
    flex: 1,
    justifyContent: 'flex-end',
  },
  nutritionSheet: {
    borderTopLeftRadius: BorderRadius.xxxl,
    borderTopRightRadius: BorderRadius.xxxl,
    maxHeight: '85%',
    ...Shadow.lg,
  },
  nutritionSheetHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: Spacing.lg,
    paddingTop: Spacing.lg,
    paddingBottom: Spacing.md,
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
  },
  nutritionSheetTitle: { ...Typography.h2, fontWeight: '700' },

  nutritionSheetBody: {
    paddingHorizontal: Spacing.lg,
    paddingVertical: Spacing.lg,
  },
  nutritionSheetSubtitle: { ...Typography.body, marginBottom: Spacing.md },

  nutritionFactsList: {
    borderTopWidth: 2,
    borderBottomWidth: 2,
    borderTopColor: '#ccc',
    borderBottomColor: '#ccc',
  },
  nutritionFactRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: Spacing.md,
    borderBottomWidth: 1,
  },
  nutritionFactLabel: { ...Typography.body, fontWeight: '500', flex: 1 },
  nutritionFactValue: { ...Typography.bodyMedium, fontWeight: '700' },

  disclaimerText: { ...Typography.caption, textAlign: 'center', lineHeight: 18 },

  closeBtn: {
    marginHorizontal: Spacing.lg,
    marginBottom: Spacing.lg,
    paddingVertical: Spacing.lg,
    borderRadius: BorderRadius.xl,
    alignItems: 'center',
  },
  closeBtnText: { ...Typography.bodyMedium, color: '#fff', fontWeight: '700' },

  // Nutrition Display
  nutritionCard: {},
  nutritionGrid: { flexDirection: 'row', justifyContent: 'space-between' },
  nutritionItem: { alignItems: 'center', flex: 1, gap: 2 },
  nutritionValue: { ...Typography.h2 },
  nutritionUnit: { ...Typography.small, textTransform: 'lowercase' },
  nutritionLabel: { ...Typography.caption, marginTop: 2 },
  resultActions: { flexDirection: 'row', gap: Spacing.md, marginTop: Spacing.xl },
});
