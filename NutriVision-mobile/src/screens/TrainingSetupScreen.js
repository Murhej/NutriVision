import React, { useState } from 'react';
import { KeyboardAvoidingView, Platform, ScrollView, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Button } from '../components';
import { BorderRadius, Shadow, Spacing, Typography } from '../theme';
import { useTheme } from '../theme/ThemeContext';

const H_PADDING = 18;
const PREFER_NOT_TO_ANSWER = 'Prefer Not to Answer';

const EQUIPMENT_OPTIONS = [
  {
    title: 'Full Gym',
    subtitle: 'Access to complete gym facilities',
  },
  {
    title: 'Home Gym',
    subtitle: 'Personal gym setup at home',
  },
  {
    title: 'Dumbbells Only',
    subtitle: 'Just dumbbells available',
  },
  {
    title: 'Resistance Bands',
    subtitle: 'Bands and light equipment',
  },
  {
    title: 'Bodyweight Only',
    subtitle: 'No equipment needed',
  },
  {
    title: 'No Equipment',
    subtitle: 'Training from scratch',
  },
  {
    title: 'Outdoor Only',
    subtitle: 'Parks, runs, outdoor spaces',
  },
  {
    title: 'Sports-Based Activity',
    subtitle: 'Team sports / recreational activity',
  },
  {
    title: PREFER_NOT_TO_ANSWER,
    subtitle: 'Skip this question',
  },
];

const TRAINING_STYLE_OPTIONS = [
  'Strength Training',
  'Hypertrophy / Bodybuilding',
  'Fat Loss Training',
  'Cardio',
  'HIIT',
  'Mobility / Flexibility',
  'Sports Performance',
  'Beginner Fitness',
  'Mixed Training',
];

export default function TrainingSetupScreen({ navigation, route }) {
  const { colors } = useTheme();
  const insets = useSafeAreaInsets();
  const [selectedEquipment, setSelectedEquipment] = useState([]);
  const [selectedStyles, setSelectedStyles] = useState([]);

  const toggleEquipment = (option) => {
    setSelectedEquipment((prev) => {
      const isSelected = prev.includes(option);

      if (isSelected) {
        return prev.filter((item) => item !== option);
      }

      if (option === PREFER_NOT_TO_ANSWER) {
        return [PREFER_NOT_TO_ANSWER];
      }

      return [...prev.filter((item) => item !== PREFER_NOT_TO_ANSWER), option];
    });
  };

  const toggleStyle = (style) => {
    setSelectedStyles((prev) =>
      prev.includes(style) ? prev.filter((item) => item !== style) : [...prev, style],
    );
  };

  const handleContinue = () => {
    navigation.navigate('ProfileReview', {
      onboardingData: {
        ...(route?.params?.onboardingData || {}),
        trainingEquipment: selectedEquipment,
        trainingStyles: selectedStyles,
      },
    });
  };

  return (
    <KeyboardAvoidingView
      style={[styles.container, { backgroundColor: colors.background }]}
      behavior={Platform.OS === 'ios' ? 'padding' : undefined}
    >
      <View
        style={[
          styles.stickyTop,
          {
            backgroundColor: colors.background,
            paddingTop: insets.top + 8,
            borderBottomColor: colors.borderLight,
          },
        ]}
      >
        <TouchableOpacity
          style={[styles.backButton, { backgroundColor: colors.surface, borderColor: colors.border }]}
          onPress={() => navigation.goBack()}
          activeOpacity={0.8}
        >
          <Text style={[styles.backText, { color: colors.textSecondary }]}>{'<'}</Text>
        </TouchableOpacity>

        <View style={[styles.progressTrack, { backgroundColor: colors.borderLight }]}>
          <View style={[styles.progressFill, { backgroundColor: colors.primary }]} />
        </View>
      </View>

      <ScrollView
        style={styles.scrollView}
        contentContainerStyle={[styles.content, { paddingBottom: 178 + insets.bottom }]}
        showsVerticalScrollIndicator={false}
      >
        <View style={styles.headerRow}>
          <Text style={[styles.title, { color: colors.text }]}>Training Setup</Text>
          <View style={[styles.optionalBadge, { backgroundColor: colors.primarySoft, borderColor: colors.primaryLight }]}>
            <Text style={[styles.optionalText, { color: colors.primaryDark }]}>Optional</Text>
          </View>
        </View>

        <Text style={[styles.subtitle, { color: colors.textSecondary }]}>What equipment do you have access to?</Text>
        <Text style={[styles.helperGreen, { color: colors.primary }]}>Select all that apply</Text>

        <View style={styles.equipmentList}>
          {EQUIPMENT_OPTIONS.map((option) => {
            const isSelected = selectedEquipment.includes(option.title);
            return (
              <TouchableOpacity
                key={option.title}
                style={[
                  styles.optionCard,
                  {
                    borderColor: isSelected ? colors.primary : colors.border,
                    backgroundColor: isSelected ? colors.primarySoft : colors.surface,
                    shadowColor: colors.shadowColor,
                    shadowOpacity: colors.shadowOpacity,
                  },
                ]}
                onPress={() => toggleEquipment(option.title)}
                activeOpacity={0.88}
              >
                <View style={styles.optionTextWrap}>
                  <Text style={[styles.optionTitle, { color: isSelected ? colors.primaryDark : colors.text }]}>{option.title}</Text>
                  <Text style={[styles.optionSubtitle, { color: colors.textSecondary }]}>{option.subtitle}</Text>
                </View>

                <View
                  style={[
                    styles.indicatorOuter,
                    {
                      borderColor: isSelected ? colors.primary : colors.border,
                      backgroundColor: isSelected ? colors.primarySoft : colors.surface,
                    },
                  ]}
                >
                  {isSelected ? <View style={[styles.indicatorInner, { backgroundColor: colors.primary }]} /> : null}
                </View>
              </TouchableOpacity>
            );
          })}
        </View>

        <Text style={[styles.stylesLabel, { color: colors.textSecondary }]}>Preferred training style (optional)</Text>
        <View style={styles.stylesWrap}>
          {TRAINING_STYLE_OPTIONS.map((style) => {
            const isSelected = selectedStyles.includes(style);
            return (
              <TouchableOpacity
                key={style}
                style={[
                  styles.styleChip,
                  {
                    borderColor: isSelected ? colors.primary : colors.border,
                    backgroundColor: isSelected ? colors.primarySoft : colors.surface,
                  },
                ]}
                onPress={() => toggleStyle(style)}
                activeOpacity={0.85}
              >
                <Text style={[styles.styleChipText, { color: isSelected ? colors.primaryDark : colors.textSecondary }]}>{style}</Text>
              </TouchableOpacity>
            );
          })}
        </View>
      </ScrollView>

      <View
        style={[
          styles.stickyBottom,
          {
            paddingBottom: insets.bottom + Spacing.md,
            backgroundColor: colors.background,
            borderTopColor: colors.borderLight,
          },
        ]}
      >
        <Button title="Continue" size="lg" onPress={handleContinue} style={styles.cta} />
        <TouchableOpacity style={styles.skipButton} onPress={handleContinue} activeOpacity={0.8}>
          <Text style={[styles.skipText, { color: colors.textSecondary }]}>Skip for now</Text>
        </TouchableOpacity>
      </View>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  stickyTop: {
    paddingHorizontal: H_PADDING,
    paddingBottom: 10,
    borderBottomWidth: 1,
  },
  backButton: {
    width: 44,
    height: 44,
    borderRadius: 22,
    borderWidth: 1,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 10,
    ...Shadow.sm,
  },
  backText: {
    ...Typography.h2,
    lineHeight: 20,
  },
  progressTrack: {
    height: 6,
    borderRadius: BorderRadius.full,
    overflow: 'hidden',
  },
  progressFill: {
    width: '98%',
    height: '100%',
    borderRadius: BorderRadius.full,
  },
  scrollView: {
    flex: 1,
  },
  content: {
    paddingHorizontal: H_PADDING,
    paddingTop: 18,
  },
  headerRow: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    justifyContent: 'space-between',
    gap: Spacing.md,
  },
  title: {
    ...Typography.hero,
    fontSize: 24,
    flex: 1,
  },
  optionalBadge: {
    borderWidth: 1,
    borderRadius: BorderRadius.full,
    paddingHorizontal: Spacing.md,
    paddingVertical: 6,
    marginTop: 3,
  },
  optionalText: {
    ...Typography.captionMedium,
  },
  subtitle: {
    ...Typography.body,
    fontSize: 15,
    marginTop: 8,
    marginBottom: 8,
  },
  helperGreen: {
    ...Typography.h3,
    marginBottom: 14,
    fontSize: 18,
  },
  equipmentList: {
    gap: 12,
  },
  optionCard: {
    borderWidth: 1,
    borderRadius: BorderRadius.lg,
    minHeight: 102,
    paddingHorizontal: Spacing.lg,
    paddingVertical: 14,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: Spacing.md,
    ...Shadow.sm,
  },
  optionTextWrap: {
    flex: 1,
  },
  optionTitle: {
    ...Typography.h3,
    fontSize: 17,
    marginBottom: 4,
  },
  optionSubtitle: {
    ...Typography.body,
    fontSize: 14,
    lineHeight: 21,
  },
  indicatorOuter: {
    width: 34,
    height: 34,
    borderRadius: 17,
    borderWidth: 1.5,
    alignItems: 'center',
    justifyContent: 'center',
  },
  indicatorInner: {
    width: 16,
    height: 16,
    borderRadius: 8,
  },
  stylesLabel: {
    ...Typography.h3,
    marginTop: Spacing.xl,
    marginBottom: Spacing.md,
  },
  stylesWrap: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
  },
  styleChip: {
    borderWidth: 1,
    borderRadius: BorderRadius.full,
    paddingVertical: 12,
    paddingHorizontal: 18,
    minHeight: 46,
    justifyContent: 'center',
  },
  styleChipText: {
    ...Typography.bodyMedium,
    fontSize: 15,
  },
  stickyBottom: {
    borderTopWidth: 1,
    paddingHorizontal: H_PADDING,
    paddingTop: 10,
  },
  cta: {
    width: '100%',
  },
  skipButton: {
    alignItems: 'center',
    justifyContent: 'center',
    paddingTop: 12,
    paddingBottom: Spacing.xs,
  },
  skipText: {
    ...Typography.bodyMedium,
  },
});
