import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, ScrollView, TouchableOpacity, Image, Alert, ActivityIndicator } from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { useNavigation } from '@react-navigation/native';
import { useTheme } from '../theme/ThemeContext';
import { Typography, Spacing, BorderRadius, Shadow } from '../theme';
import { Card, Button } from '../components';
import { Camera, Upload } from 'lucide-react-native';
import { apiClient } from '../api/client';

export default function ScanScreen() {
  const { colors } = useTheme();
  const navigation = useNavigation();
  const [step, setStep] = useState('upload'); // 'upload' | 'results'
  const [isUploading, setIsUploading] = useState(false);
  const [isFetchingNutrition, setIsFetchingNutrition] = useState(false);
  
  const [imageUri, setImageUri] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [selectedPrediction, setSelectedPrediction] = useState(0);
  const [nutrition, setNutrition] = useState(null);

  const processImage = async (uri) => {
    setImageUri(uri);
    setStep('results');
    setIsUploading(true);
    setPredictions([]);
    setNutrition(null);
    
    try {
      const result = await apiClient.uploadImage('/predict', uri);
      if (result.predictions && result.predictions.length > 0) {
        // Prepare display labels
        const enhancedPredictions = result.predictions.map(p => ({
          label: p.class.replace(/_/g, ' '),
          rawClass: p.class,
          confidence: Math.round(p.confidence),
          emoji: '🍽️'
        }));
        setPredictions(enhancedPredictions);
        setSelectedPrediction(0);
        fetchNutrition(enhancedPredictions[0].rawClass);
      } else {
        Alert.alert('Analysis Failed', 'Could not identify food in this image.');
        setStep('upload');
      }
    } catch (err) {
      console.warn('API /predict failed', err);
      Alert.alert(
        'Model Not Loaded!',
        'Your PyTorch model is missing. Please run `python main.py train` on your computer first to generate the brain!'
      );
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
        setNutrition(null);
      }
    } catch (error) {
       console.warn('Failed to fetch nutrition:', error);
       setNutrition(null);
    } finally {
      setIsFetchingNutrition(false);
    }
  };

  const handleSelectPrediction = (index) => {
    if (index === selectedPrediction) return;
    setSelectedPrediction(index);
    const pred = predictions[index];
    if (pred) {
      fetchNutrition(pred.rawClass);
    }
  };

  const handleTakePhoto = async () => {
    try {
      const { status } = await ImagePicker.requestCameraPermissionsAsync();
      if (status !== 'granted') {
        Alert.alert('Permission needed', 'Sorry, we need camera permissions to make this work!');
        return;
      }
      const result = await ImagePicker.launchCameraAsync({ allowsEditing: true, quality: 0.8 });
      if (!result.canceled) {
        processImage(result.assets[0].uri);
      }
    } catch (error) {
      Alert.alert('Camera Unavailable', 'Try uploading from gallery instead.');
    }
  };

  const handleUploadPhoto = async () => {
    try {
      const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
      if (status !== 'granted') {
        Alert.alert('Permission needed', 'Sorry, we need camera roll permissions!');
        return;
      }
      const result = await ImagePicker.launchImageLibraryAsync({ allowsEditing: true, quality: 0.8 });
      if (!result.canceled) {
        processImage(result.assets[0].uri);
      }
    } catch (error) {
      Alert.alert('Gallery Unavailable', 'Could not open photo gallery.');
    }
  };

  const handleLogMeal = async () => {
    if (!predictions[selectedPrediction] || !nutrition) return;
    try {
      await apiClient.post('/map/log', {
        food_label: predictions[selectedPrediction].rawClass,
        display_name: predictions[selectedPrediction].label,
        portion_id: 'medium',
        portion_multiplier: 1.0,
        nutrition: nutrition,
        prediction: predictions[selectedPrediction],
        source: 'camera'
      });
      // Head back to dashboard to see new meal!
      handleReset();
      navigation.navigate('Dashboard');
    } catch (err) {
      Alert.alert('Save Failed', 'Could not log meal to the server.');
    }
  };

  const handleReset = () => {
    setStep('upload');
    setImageUri(null);
    setPredictions([]);
    setNutrition(null);
  };

  return (
    <ScrollView style={[styles.container, { backgroundColor: colors.background }]} contentContainerStyle={styles.content} showsVerticalScrollIndicator={false}>
      {step === 'upload' && (
        <>
          <Card style={styles.uploadCard}>
            <View style={[styles.uploadArea, { borderColor: colors.border, backgroundColor: colors.surfaceSecondary }]}>
              <View style={[styles.uploadIconCircle, { backgroundColor: colors.primarySoft }]}>
                <Camera size={32} color={colors.primary} />
              </View>
              <Text style={[styles.uploadTitle, { color: colors.text }]}>Scan Your Meal</Text>
              <Text style={[styles.uploadSubtitle, { color: colors.textSecondary }]}>
                Take a photo or upload an image to identify your food and get nutrition info
              </Text>
            </View>
          </Card>
          <View style={styles.actionRow}>
            <TouchableOpacity style={[styles.actionBtn, { backgroundColor: colors.primary }]} onPress={handleTakePhoto} activeOpacity={0.85}>
              <Camera size={22} color="#fff" />
              <Text style={styles.actionBtnText}>Take Photo</Text>
            </TouchableOpacity>
            <TouchableOpacity style={[styles.actionBtn, { backgroundColor: colors.surface, borderColor: colors.border, borderWidth: 1 }]} onPress={handleUploadPhoto} activeOpacity={0.85}>
              <Upload size={22} color={colors.primary} />
              <Text style={[styles.actionBtnText, { color: colors.primary }]}>Upload</Text>
            </TouchableOpacity>
          </View>
          <Card style={styles.tipsCard}>
            <Text style={[styles.tipsTitle, { color: colors.text }]}>📌 Tips for best results</Text>
            {['Place your food on a clean surface', 'Ensure good lighting', 'Capture the entire plate', 'Take the photo from above'].map((tip, idx) => (
              <View key={idx} style={styles.tipRow}>
                <Text style={[styles.tipBullet, { color: colors.primary }]}>•</Text>
                <Text style={[styles.tipText, { color: colors.textSecondary }]}>{tip}</Text>
              </View>
            ))}
          </Card>
        </>
      )}

      {step === 'results' && (
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
            <View style={{ padding: 40, alignItems: 'center' }}>
              <ActivityIndicator size="large" color={colors.primary} />
              <Text style={{ marginTop: 16, color: colors.textSecondary }}>Running AI Vision Model...</Text>
            </View>
          ) : (
            <>
              <Text style={[styles.sectionTitle, { color: colors.text }]}>What did you eat?</Text>
              <Text style={[styles.sectionSubtitle, { color: colors.textSecondary }]}>Select the best match from AI predictions</Text>

              <View style={styles.predictionsCol}>
                {predictions.map((pred, index) => (
                  <TouchableOpacity
                    key={index}
                    style={[styles.predictionCard, { backgroundColor: colors.surface, borderColor: index === selectedPrediction ? colors.primary : colors.border, borderWidth: index === selectedPrediction ? 2 : 1, ...Shadow.sm }]}
                    onPress={() => handleSelectPrediction(index)}
                    activeOpacity={0.7}
                  >
                    <View style={[styles.predEmoji, { backgroundColor: colors.primarySoft }]}><Text style={{ fontSize: 22 }}>{pred.emoji}</Text></View>
                    <View style={styles.predInfo}>
                      <Text style={[styles.predLabel, { color: colors.text }]}>{pred.label}</Text>
                      <Text style={[styles.predConf, { color: colors.textSecondary }]}>{pred.confidence}% confidence</Text>
                    </View>
                    {index === selectedPrediction && (
                      <View style={[styles.checkMark, { backgroundColor: colors.primary }]}><Text style={styles.checkText}>✓</Text></View>
                    )}
                  </TouchableOpacity>
                ))}
              </View>

              <Text style={[styles.sectionTitle, { color: colors.text, marginTop: Spacing.xl }]}>Nutrition Breakdown</Text>
              <Card style={styles.nutritionCard}>
                {isFetchingNutrition ? (
                  <View style={{ padding: 20, alignItems: 'center' }}><ActivityIndicator color={colors.primary} /></View>
                ) : nutrition ? (
                  <View style={styles.nutritionGrid}>
                    {[
                      { label: 'Calories', value: nutrition.calories, unit: 'kcal', color: colors.calories },
                      { label: 'Protein', value: nutrition.protein_g, unit: 'g', color: colors.protein },
                      { label: 'Carbs', value: nutrition.carbs_g, unit: 'g', color: colors.carbs },
                      { label: 'Fat', value: nutrition.fat_g, unit: 'g', color: colors.fat },
                    ].map((item, idx) => (
                      <View key={idx} style={styles.nutritionItem}>
                        <Text style={[styles.nutritionValue, { color: item.color }]}>{item.value}</Text>
                        <Text style={[styles.nutritionUnit, { color: colors.textTertiary }]}>{item.unit}</Text>
                        <Text style={[styles.nutritionLabel, { color: colors.textSecondary }]}>{item.label}</Text>
                      </View>
                    ))}
                  </View>
                ) : (
                  <Text style={{ textAlign: 'center', color: colors.textSecondary, padding: 20 }}>No nutrition data found for this item.</Text>
                )}
              </Card>

              <View style={styles.resultActions}>
                <Button title="Add to Log" onPress={handleLogMeal} size="lg" style={{ flex: 1 }} disabled={!nutrition || isFetchingNutrition} />
                <Button title="Retake" variant="outline" onPress={handleReset} size="lg" style={{ flex: 1 }} />
              </View>
            </>
          )}
        </>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  content: { padding: Spacing.lg, paddingBottom: Spacing.xxxl * 2 },
  uploadCard: { padding: 0, overflow: 'hidden', marginBottom: Spacing.lg },
  uploadArea: { alignItems: 'center', justifyContent: 'center', padding: Spacing.xxxl, borderWidth: 2, borderStyle: 'dashed', borderRadius: BorderRadius.lg, margin: Spacing.md, gap: Spacing.md },
  uploadIconCircle: { width: 64, height: 64, borderRadius: 32, alignItems: 'center', justifyContent: 'center', marginBottom: Spacing.sm },
  uploadTitle: { ...Typography.h2, textAlign: 'center' },
  uploadSubtitle: { ...Typography.body, textAlign: 'center', maxWidth: 260 },
  actionRow: { flexDirection: 'row', gap: Spacing.md, marginBottom: Spacing.xl },
  actionBtn: { flex: 1, flexDirection: 'row', alignItems: 'center', justifyContent: 'center', padding: Spacing.lg, borderRadius: BorderRadius.xl, gap: Spacing.sm },
  actionBtnText: { ...Typography.bodyMedium, color: '#fff', fontWeight: '700' },
  tipsCard: { gap: Spacing.sm },
  tipsTitle: { ...Typography.bodyMedium, fontWeight: '700', marginBottom: Spacing.xs },
  tipRow: { flexDirection: 'row', gap: Spacing.sm, alignItems: 'flex-start' },
  tipBullet: { fontSize: 16, lineHeight: 20 },
  tipText: { ...Typography.caption, lineHeight: 20, flex: 1 },
  previewCard: { padding: 0, overflow: 'hidden', marginBottom: Spacing.lg },
  previewImage: { height: 220, alignItems: 'center', justifyContent: 'center' },
  confidenceBadge: { position: 'absolute', top: Spacing.md, right: Spacing.md, paddingHorizontal: Spacing.md, paddingVertical: Spacing.xs, borderRadius: BorderRadius.full },
  confidenceText: { color: '#fff', ...Typography.captionMedium },
  sectionTitle: { ...Typography.h3, marginBottom: Spacing.xs },
  sectionSubtitle: { ...Typography.caption, marginBottom: Spacing.lg },
  predictionsCol: { gap: Spacing.md, marginBottom: Spacing.md },
  predictionCard: { flexDirection: 'row', alignItems: 'center', padding: Spacing.md, borderRadius: BorderRadius.lg, gap: Spacing.md, ...Shadow.sm },
  predEmoji: { width: 44, height: 44, borderRadius: 22, alignItems: 'center', justifyContent: 'center' },
  predInfo: { flex: 1, gap: 2 },
  predLabel: { ...Typography.bodyMedium, textTransform: 'capitalize' },
  predConf: { ...Typography.caption },
  checkMark: { width: 28, height: 28, borderRadius: 14, alignItems: 'center', justifyContent: 'center' },
  checkText: { color: '#fff', fontWeight: '700', fontSize: 14 },
  nutritionCard: {},
  nutritionGrid: { flexDirection: 'row', justifyContent: 'space-between' },
  nutritionItem: { alignItems: 'center', flex: 1, gap: 2 },
  nutritionValue: { ...Typography.h2 },
  nutritionUnit: { ...Typography.small, textTransform: 'lowercase' },
  nutritionLabel: { ...Typography.caption, marginTop: 2 },
  resultActions: { flexDirection: 'row', gap: Spacing.md, marginTop: Spacing.xl },
});
