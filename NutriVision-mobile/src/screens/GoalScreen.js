import React, { useMemo, useState } from 'react';
import { KeyboardAvoidingView, Platform, ScrollView, StyleSheet, Text, TextInput, TouchableOpacity, View } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Button } from '../components';
import { BorderRadius, Shadow, Spacing, Typography } from '../theme';
import { useTheme } from '../theme/ThemeContext';

const H_PADDING = 18;

const GOALS = [
  { title: 'Lose Weight', subtitle: 'Reduce body fat safely' },
  { title: 'Build Muscle', subtitle: 'Gain lean strength' },
  { title: 'Maintain Weight', subtitle: 'Stay at your current weight' },
  { title: 'Body Recomposition', subtitle: 'Lose fat and gain muscle' },
  { title: 'Improve Performance', subtitle: 'Recover and perform better' },
  { title: 'Increase Energy', subtitle: 'Feel better every day' },
  { title: 'Improve Heart Health', subtitle: 'Support cardiovascular health' },
  { title: 'Manage Blood Sugar', subtitle: 'Keep levels in healthy range' },
  { title: 'Improve Digestion', subtitle: 'Support gut comfort' },
  { title: 'Boost Immunity', subtitle: 'Support immune function' },
  { title: 'Build Consistency', subtitle: 'Create healthy routines' },
  { title: 'Improve Sleep', subtitle: 'Eat better for better rest' },
];

export default function GoalScreen({ navigation, route }) {
  const { colors } = useTheme();
  const insets = useSafeAreaInsets();
  const [selectedGoals, setSelectedGoals] = useState([]);
  const [customGoal, setCustomGoal] = useState('');
  const [hasTriedToContinue, setHasTriedToContinue] = useState(false);

  const canContinue = useMemo(() => selectedGoals.length > 0, [selectedGoals]);
  const selectionError = hasTriedToContinue && !canContinue ? 'Please select at least one option to continue.' : '';

  const toggleGoal = (goalTitle) => {
    setSelectedGoals((prev) =>
      prev.includes(goalTitle) ? prev.filter((goal) => goal !== goalTitle) : [...prev, goalTitle],
    );
  };

  const handleContinue = () => {
    setHasTriedToContinue(true);
    if (!canContinue) return;

    navigation.navigate('Challenge', {
      onboardingData: {
        ...(route?.params?.onboardingData || {}),
        selectedGoals,
        customGoal: customGoal.trim(),
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
        contentContainerStyle={[styles.content, { paddingBottom: 152 + insets.bottom }]}
        showsVerticalScrollIndicator={false}
        keyboardShouldPersistTaps="handled"
      >
        <Text style={[styles.title, { color: colors.text }]}>What's your main goal?</Text>
        <Text style={[styles.subtitle, { color: colors.textSecondary }]}>Select all that apply so we can personalize your plan.</Text>
        <Text style={[styles.helperText, { color: colors.textTertiary }]}>Choose one or more options.</Text>

        <View style={styles.grid}>
          {GOALS.map((goal) => {
            const isSelected = selectedGoals.includes(goal.title);
            return (
              <TouchableOpacity
                key={goal.title}
                style={[
                  styles.goalCard,
                  {
                    borderColor: isSelected ? colors.primary : selectionError ? colors.danger : colors.border,
                    backgroundColor: isSelected ? colors.primarySoft : colors.surface,
                    shadowColor: colors.shadowColor,
                    shadowOpacity: colors.shadowOpacity,
                  },
                ]}
                onPress={() => toggleGoal(goal.title)}
                activeOpacity={0.88}
              >
                <View style={styles.goalTopRow}>
                  <View
                    style={[
                      styles.checkboxOuter,
                      {
                        borderColor: isSelected ? colors.primary : colors.border,
                        backgroundColor: isSelected ? colors.primary : colors.surface,
                      },
                    ]}
                  >
                    {isSelected ? <Text style={styles.checkboxMark}>✓</Text> : null}
                  </View>
                </View>
                <Text style={[styles.goalTitle, { color: isSelected ? colors.primaryDark : colors.text }]}>{goal.title}</Text>
                <Text style={[styles.goalSubtitle, { color: colors.textSecondary }]}>{goal.subtitle}</Text>
              </TouchableOpacity>
            );
          })}
        </View>

        {selectionError ? <Text style={[styles.errorText, { color: colors.danger }]}>{selectionError}</Text> : null}

        <Text style={[styles.label, { color: colors.textSecondary }]}>Other goal (optional)</Text>
        <TextInput
          style={[styles.input, { borderColor: colors.border, color: colors.text, backgroundColor: colors.surface }]}
          placeholder="Enter your custom goal"
          placeholderTextColor={colors.textTertiary}
          value={customGoal}
          onChangeText={setCustomGoal}
        />
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
        <Button title="Continue" size="lg" onPress={handleContinue} disabled={!canContinue} style={styles.cta} />
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
  scrollView: {
    flex: 1,
  },
  content: {
    paddingHorizontal: H_PADDING,
    paddingTop: 18,
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
    width: '36%',
    height: '100%',
    borderRadius: BorderRadius.full,
  },
  title: {
    ...Typography.hero,
    fontSize: 48 / 2,
    marginBottom: 6,
  },
  subtitle: {
    ...Typography.body,
    marginBottom: 4,
    fontSize: 15,
  },
  helperText: {
    ...Typography.caption,
    marginBottom: 14,
  },
  grid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: Spacing.sm,
  },
  goalCard: {
    width: '48%',
    borderWidth: 1,
    borderRadius: BorderRadius.lg,
    paddingHorizontal: Spacing.md,
    paddingVertical: 12,
    minHeight: 116,
    ...Shadow.sm,
  },
  goalTopRow: {
    flexDirection: 'row',
    justifyContent: 'flex-end',
    alignItems: 'center',
    marginBottom: 6,
  },
  checkboxOuter: {
    width: 22,
    height: 22,
    borderRadius: 11,
    borderWidth: 1.5,
    alignItems: 'center',
    justifyContent: 'center',
  },
  checkboxMark: {
    color: '#ffffff',
    fontSize: 12,
    fontWeight: '800',
  },
  goalTitle: {
    ...Typography.bodyMedium,
    fontSize: 15,
    marginBottom: 4,
  },
  goalSubtitle: {
    ...Typography.caption,
    fontSize: 14,
    lineHeight: 20,
  },
  label: {
    ...Typography.bodyMedium,
    marginTop: 14,
    marginBottom: 8,
  },
  input: {
    borderWidth: 1,
    borderRadius: BorderRadius.lg,
    paddingHorizontal: Spacing.lg,
    paddingVertical: 14,
    fontSize: 15,
  },
  errorText: {
    ...Typography.captionMedium,
    marginTop: 10,
  },
  stickyBottom: {
    borderTopWidth: 1,
    paddingHorizontal: H_PADDING,
    paddingTop: 10,
  },
  cta: {
    width: '100%',
  },
});
