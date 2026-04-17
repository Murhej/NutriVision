import React, { useMemo, useState } from 'react';
import { KeyboardAvoidingView, Platform, ScrollView, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Button } from '../components';
import { BorderRadius, Shadow, Spacing, Typography } from '../theme';
import { useTheme } from '../theme/ThemeContext';

const H_PADDING = 18;

const ACTIVITY_OPTIONS = [
  {
    title: 'Mostly Sitting at Home',
    subtitle: 'Desk work, minimal movement',
  },
  {
    title: 'Desk Job / Sitting All Day',
    subtitle: 'Office work with little walking',
  },
  {
    title: 'Light Walking During Day',
    subtitle: 'Some walking, mostly sitting',
  },
  {
    title: 'Retail / Standing Job',
    subtitle: 'On your feet most of the day',
  },
  {
    title: 'Active Job',
    subtitle: 'Regular walking and light lifting',
  },
  {
    title: 'Construction / Manual Labor',
    subtitle: 'Heavy physical work daily',
  },
  {
    title: 'Very Physically Demanding',
    subtitle: 'Intense physical activity all day',
  },
];

const DURATION_OPTIONS = ['1-2h', '3-5h', '5-7h', '7+h'];

export default function ActivityLevelScreen({ navigation, route }) {
  const { colors } = useTheme();
  const insets = useSafeAreaInsets();
  const [selectedActivity, setSelectedActivity] = useState('');
  const [selectedDuration, setSelectedDuration] = useState('');
  const [hasTriedContinue, setHasTriedContinue] = useState(false);

  const canContinue = useMemo(() => Boolean(selectedActivity) && Boolean(selectedDuration), [selectedActivity, selectedDuration]);

  const handleContinue = () => {
    setHasTriedContinue(true);
    if (!canContinue) return;

    navigation.navigate('TrainingSetup', {
      onboardingData: {
        ...(route?.params?.onboardingData || {}),
        activityLevel: selectedActivity,
        weeklyExerciseDuration: selectedDuration,
      },
    });
  };

  const activityError = hasTriedContinue && !selectedActivity ? 'Select your activity level.' : '';
  const durationError = hasTriedContinue && !selectedDuration ? 'Select your weekly exercise duration.' : '';

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
        contentContainerStyle={[styles.content, { paddingBottom: 146 + insets.bottom }]}
        showsVerticalScrollIndicator={false}
      >
        <Text style={[styles.title, { color: colors.text }]}>Exercise Habits</Text>
        <Text style={[styles.subtitle, { color: colors.textSecondary }]}>How often do you work out?</Text>
        <Text style={[styles.requiredText, { color: colors.primary }]}>This step is required</Text>

        <View style={styles.list}>
          {ACTIVITY_OPTIONS.map((option) => {
            const isSelected = selectedActivity === option.title;
            return (
              <TouchableOpacity
                key={option.title}
                style={[
                  styles.optionCard,
                  {
                    borderColor: isSelected ? colors.primary : activityError ? colors.danger : colors.border,
                    backgroundColor: isSelected ? colors.primarySoft : colors.surface,
                    shadowColor: colors.shadowColor,
                    shadowOpacity: colors.shadowOpacity,
                  },
                ]}
                onPress={() => setSelectedActivity(option.title)}
                activeOpacity={0.88}
              >
                <View style={styles.optionTextWrap}>
                  <Text style={[styles.optionTitle, { color: isSelected ? colors.primaryDark : colors.text }]}>{option.title}</Text>
                  <Text style={[styles.optionSubtitle, { color: colors.textSecondary }]}>{option.subtitle}</Text>
                </View>

                <View
                  style={[
                    styles.radioOuter,
                    {
                      borderColor: isSelected ? colors.primary : colors.border,
                      backgroundColor: isSelected ? colors.primarySoft : colors.surface,
                    },
                  ]}
                >
                  {isSelected ? <View style={[styles.radioInner, { backgroundColor: colors.primary }]} /> : null}
                </View>
              </TouchableOpacity>
            );
          })}
        </View>

        {activityError ? <Text style={[styles.errorText, { color: colors.danger }]}>{activityError}</Text> : null}

        <Text style={[styles.durationTitle, { color: colors.textSecondary }]}>Weekly exercise duration</Text>
        <View style={styles.durationRow}>
          {DURATION_OPTIONS.map((duration) => {
            const isSelected = selectedDuration === duration;
            return (
              <TouchableOpacity
                key={duration}
                style={[
                  styles.durationChip,
                  {
                    borderColor: isSelected ? colors.primary : durationError ? colors.danger : colors.border,
                    backgroundColor: isSelected ? colors.primarySoft : colors.surface,
                  },
                ]}
                onPress={() => setSelectedDuration(duration)}
                activeOpacity={0.86}
              >
                <Text style={[styles.durationChipText, { color: isSelected ? colors.primaryDark : colors.textSecondary }]}>{duration}</Text>
              </TouchableOpacity>
            );
          })}
        </View>

        {durationError ? <Text style={[styles.errorText, { color: colors.danger }]}>{durationError}</Text> : null}
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
        <Button title="Continue" size="lg" onPress={handleContinue} style={[styles.cta, !canContinue && styles.ctaBlocked]} />
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
    width: '92%',
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
  title: {
    ...Typography.hero,
    fontSize: 24,
    marginBottom: 8,
  },
  subtitle: {
    ...Typography.body,
    fontSize: 15,
    marginBottom: 8,
  },
  requiredText: {
    ...Typography.h3,
    marginBottom: 14,
    fontSize: 18,
  },
  list: {
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
    fontSize: 20 / 1.2,
    marginBottom: 4,
  },
  optionSubtitle: {
    ...Typography.body,
    fontSize: 14,
    lineHeight: 21,
  },
  radioOuter: {
    width: 34,
    height: 34,
    borderRadius: 17,
    borderWidth: 1.5,
    alignItems: 'center',
    justifyContent: 'center',
  },
  radioInner: {
    width: 16,
    height: 16,
    borderRadius: 8,
  },
  errorText: {
    ...Typography.captionMedium,
    marginTop: 10,
  },
  durationTitle: {
    ...Typography.h3,
    marginTop: Spacing.xl,
    marginBottom: Spacing.md,
  },
  durationRow: {
    flexDirection: 'row',
    gap: Spacing.md,
    flexWrap: 'wrap',
  },
  durationChip: {
    borderWidth: 1,
    borderRadius: BorderRadius.full,
    minWidth: 88,
    minHeight: 52,
    paddingHorizontal: Spacing.lg,
    paddingVertical: 12,
    alignItems: 'center',
    justifyContent: 'center',
  },
  durationChipText: {
    ...Typography.h3,
    fontSize: 19,
  },
  stickyBottom: {
    borderTopWidth: 1,
    paddingHorizontal: H_PADDING,
    paddingTop: 10,
  },
  cta: {
    width: '100%',
  },
  ctaBlocked: {
    opacity: 0.6,
  },
});
