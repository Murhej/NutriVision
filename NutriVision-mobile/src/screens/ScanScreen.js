import React, { useState } from 'react';
import { View, Text, StyleSheet, ScrollView, TouchableOpacity, Image } from 'react-native';
import { useTheme } from '../theme/ThemeContext';
import { Typography, Spacing, BorderRadius, Shadow } from '../theme';
import { Card, Button, ProgressBar } from '../components';
import { Camera, Upload, ImageIcon } from 'lucide-react-native';

// Mock scan result for demo
const MOCK_RESULT = {
  predictions: [
    { label: 'Grilled Chicken Salad', confidence: 92, emoji: '🥗' },
    { label: 'Caesar Salad', confidence: 78, emoji: '🥬' },
    { label: 'Greek Salad', confidence: 65, emoji: '🫒' },
  ],
  nutrition: {
    calories: 380,
    protein: 32,
    carbs: 18,
    fat: 14,
    fiber: 6,
    sugar: 4,
  },
};

export default function ScanScreen() {
  const { colors } = useTheme();
  const [step, setStep] = useState('upload'); // 'upload' | 'results'
  const [selectedPrediction, setSelectedPrediction] = useState(0);

  const handleMockScan = () => {
    setStep('results');
    setSelectedPrediction(0);
  };

  const handleReset = () => {
    setStep('upload');
  };

  return (
    <ScrollView
      style={[styles.container, { backgroundColor: colors.background }]}
      contentContainerStyle={styles.content}
      showsVerticalScrollIndicator={false}
    >
      {step === 'upload' && (
        <>
          {/* Upload Area */}
          <Card style={styles.uploadCard}>
            <View style={[styles.uploadArea, { borderColor: colors.border, backgroundColor: colors.surfaceSecondary }]}>
              <View style={[styles.uploadIconCircle, { backgroundColor: colors.primarySoft }]}>
                <Camera size={32} color={colors.primary} />
              </View>
              <Text style={[styles.uploadTitle, { color: colors.text }]}>
                Scan Your Meal
              </Text>
              <Text style={[styles.uploadSubtitle, { color: colors.textSecondary }]}>
                Take a photo or upload an image to identify your food and get nutrition info
              </Text>
            </View>
          </Card>

          {/* Action Buttons */}
          <View style={styles.actionRow}>
            <TouchableOpacity
              style={[styles.actionBtn, { backgroundColor: colors.primary }]}
              onPress={handleMockScan}
              activeOpacity={0.85}
            >
              <Camera size={22} color="#fff" />
              <Text style={styles.actionBtnText}>Take Photo</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.actionBtn, { backgroundColor: colors.surface, borderColor: colors.border, borderWidth: 1 }]}
              onPress={handleMockScan}
              activeOpacity={0.85}
            >
              <Upload size={22} color={colors.primary} />
              <Text style={[styles.actionBtnText, { color: colors.primary }]}>Upload</Text>
            </TouchableOpacity>
          </View>

          {/* Tips */}
          <Card style={styles.tipsCard}>
            <Text style={[styles.tipsTitle, { color: colors.text }]}>📌 Tips for best results</Text>
            {[
              'Place your food on a clean surface',
              'Ensure good lighting',
              'Capture the entire plate',
              'Take the photo from above',
            ].map((tip, index) => (
              <View key={index} style={styles.tipRow}>
                <Text style={[styles.tipBullet, { color: colors.primary }]}>•</Text>
                <Text style={[styles.tipText, { color: colors.textSecondary }]}>{tip}</Text>
              </View>
            ))}
          </Card>
        </>
      )}

      {step === 'results' && (
        <>
          {/* Image Preview */}
          <Card style={styles.previewCard}>
            <View style={[styles.previewImage, { backgroundColor: colors.surfaceSecondary }]}>
              <Image source={require('../../assets/food-scan.png')} style={{ width: '100%', height: '100%', resizeMode: 'cover' }} />
              <View style={[styles.confidenceBadge, { backgroundColor: colors.primary }]}>
                <Text style={styles.confidenceText}>
                  {MOCK_RESULT.predictions[selectedPrediction].confidence}% match
                </Text>
              </View>
            </View>
          </Card>

          {/* Predictions */}
          <Text style={[styles.sectionTitle, { color: colors.text }]}>What did you eat?</Text>
          <Text style={[styles.sectionSubtitle, { color: colors.textSecondary }]}>
            Select the best match from AI predictions
          </Text>

          <View style={styles.predictionsCol}>
            {MOCK_RESULT.predictions.map((pred, index) => (
              <TouchableOpacity
                key={index}
                style={[
                  styles.predictionCard,
                  {
                    backgroundColor: colors.surface,
                    borderColor: index === selectedPrediction ? colors.primary : colors.border,
                    borderWidth: index === selectedPrediction ? 2 : 1,
                    shadowColor: colors.shadowColor,
                    shadowOpacity: colors.shadowOpacity,
                  },
                ]}
                onPress={() => setSelectedPrediction(index)}
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

          {/* Nutrition Breakdown */}
          <Text style={[styles.sectionTitle, { color: colors.text, marginTop: Spacing.xl }]}>
            Nutrition Breakdown
          </Text>
          <Card style={styles.nutritionCard}>
            <View style={styles.nutritionGrid}>
              {[
                { label: 'Calories', value: MOCK_RESULT.nutrition.calories, unit: 'kcal', color: colors.calories },
                { label: 'Protein', value: MOCK_RESULT.nutrition.protein, unit: 'g', color: colors.protein },
                { label: 'Carbs', value: MOCK_RESULT.nutrition.carbs, unit: 'g', color: colors.carbs },
                { label: 'Fat', value: MOCK_RESULT.nutrition.fat, unit: 'g', color: colors.fat },
              ].map((item, index) => (
                <View key={index} style={styles.nutritionItem}>
                  <Text style={[styles.nutritionValue, { color: item.color }]}>
                    {item.value}
                  </Text>
                  <Text style={[styles.nutritionUnit, { color: colors.textTertiary }]}>
                    {item.unit}
                  </Text>
                  <Text style={[styles.nutritionLabel, { color: colors.textSecondary }]}>
                    {item.label}
                  </Text>
                </View>
              ))}
            </View>
          </Card>

          {/* Actions */}
          <View style={styles.resultActions}>
            <Button title="Add to Log" onPress={handleReset} size="lg" style={{ flex: 1 }} />
            <Button title="Retake" variant="outline" onPress={handleReset} size="lg" style={{ flex: 1 }} />
          </View>
        </>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  content: { padding: Spacing.lg, paddingBottom: Spacing.xxxl * 2 },
  uploadCard: { padding: 0, overflow: 'hidden', marginBottom: Spacing.lg },
  uploadArea: {
    alignItems: 'center',
    justifyContent: 'center',
    padding: Spacing.xxxl,
    borderWidth: 2,
    borderStyle: 'dashed',
    borderRadius: BorderRadius.lg,
    margin: Spacing.md,
    gap: Spacing.md,
  },
  uploadIconCircle: {
    width: 64,
    height: 64,
    borderRadius: 32,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: Spacing.sm,
  },
  uploadTitle: { ...Typography.h2, textAlign: 'center' },
  uploadSubtitle: { ...Typography.body, textAlign: 'center', maxWidth: 260 },
  actionRow: { flexDirection: 'row', gap: Spacing.md, marginBottom: Spacing.xl },
  actionBtn: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: Spacing.lg,
    borderRadius: BorderRadius.xl,
    gap: Spacing.sm,
  },
  actionBtnText: { ...Typography.bodyMedium, color: '#fff', fontWeight: '700' },
  tipsCard: { gap: Spacing.sm },
  tipsTitle: { ...Typography.bodyMedium, fontWeight: '700', marginBottom: Spacing.xs },
  tipRow: { flexDirection: 'row', gap: Spacing.sm, alignItems: 'flex-start' },
  tipBullet: { fontSize: 16, lineHeight: 20 },
  tipText: { ...Typography.caption, lineHeight: 20, flex: 1 },
  previewCard: { padding: 0, overflow: 'hidden', marginBottom: Spacing.lg },
  previewImage: {
    height: 220,
    alignItems: 'center',
    justifyContent: 'center',
  },
  previewEmoji: { fontSize: 80 },
  confidenceBadge: {
    position: 'absolute',
    top: Spacing.md,
    right: Spacing.md,
    paddingHorizontal: Spacing.md,
    paddingVertical: Spacing.xs,
    borderRadius: BorderRadius.full,
  },
  confidenceText: { color: '#fff', ...Typography.captionMedium },
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
  predEmoji: { width: 44, height: 44, borderRadius: 22, alignItems: 'center', justifyContent: 'center' },
  predInfo: { flex: 1, gap: 2 },
  predLabel: { ...Typography.bodyMedium },
  predConf: { ...Typography.caption },
  checkMark: {
    width: 28,
    height: 28,
    borderRadius: 14,
    alignItems: 'center',
    justifyContent: 'center',
  },
  checkText: { color: '#fff', fontWeight: '700', fontSize: 14 },
  nutritionCard: {},
  nutritionGrid: { flexDirection: 'row', justifyContent: 'space-between' },
  nutritionItem: { alignItems: 'center', flex: 1, gap: 2 },
  nutritionValue: { ...Typography.h2 },
  nutritionUnit: { ...Typography.small, textTransform: 'lowercase' },
  nutritionLabel: { ...Typography.caption, marginTop: 2 },
  resultActions: { flexDirection: 'row', gap: Spacing.md, marginTop: Spacing.xl },
});
