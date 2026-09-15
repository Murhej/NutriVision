import React, { useState } from 'react';
import { KeyboardAvoidingView, Platform, ScrollView, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Button } from '../components';
import { BorderRadius, Shadow, Spacing, Typography } from '../theme';
import { useTheme } from '../theme/ThemeContext';

const H_PADDING = 18;

const CHALLENGES = [
  'Overeating',
  'Cravings',
  'Emotional Eating',
  'Inconsistent Meals',
  'Late Night Snacking',
  'No Time to Cook',
  'Eating Out Too Often',
  'Tracking Calories',
  'Understanding Nutrition',
  'Low Energy',
  'Stress',
  'Limited Budget',
  'Family/Social Eating',
  'Staying Consistent',
  'Exercise Consistency',
  'Portion Control',
  'Skipping Meals',
  'Not Knowing What to Eat',
];

export default function ChallengeScreen({ navigation, route }) {
  const { colors } = useTheme();
  const insets = useSafeAreaInsets();
  const [selectedChallenges, setSelectedChallenges] = useState([]);

  const toggleChallenge = (challenge) => {
    setSelectedChallenges((prev) =>
      prev.includes(challenge) ? prev.filter((item) => item !== challenge) : [...prev, challenge],
    );
  };

  const finishOnboarding = () => {
    navigation.navigate('DietaryPreferences', {
      onboardingData: {
        ...(route?.params?.onboardingData || {}),
        selectedChallenges,
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
        contentContainerStyle={[styles.content, { paddingBottom: 172 + insets.bottom }]}
        showsVerticalScrollIndicator={false}
      >
        <View style={styles.headerRow}>
          <Text style={[styles.title, { color: colors.text }]}>What&apos;s your biggest challenge?</Text>
          <View style={[styles.optionalBadge, { backgroundColor: colors.primarySoft, borderColor: colors.primaryLight }]}>
            <Text style={[styles.optionalText, { color: colors.primaryDark }]}>Optional</Text>
          </View>
        </View>

        <Text style={[styles.subtitle, { color: colors.textSecondary }]}>Select all that apply. We&apos;ll help you overcome them.</Text>
        <Text style={[styles.helperText, { color: colors.textTertiary }]}>Optional. Choose any that apply.</Text>

        <View style={styles.grid}>
          {CHALLENGES.map((challenge) => {
            const isSelected = selectedChallenges.includes(challenge);
            return (
              <TouchableOpacity
                key={challenge}
                style={[
                  styles.challengeCard,
                  {
                    borderColor: isSelected ? colors.primary : colors.border,
                    backgroundColor: isSelected ? colors.primarySoft : colors.surface,
                    shadowColor: colors.shadowColor,
                    shadowOpacity: colors.shadowOpacity,
                  },
                ]}
                onPress={() => toggleChallenge(challenge)}
                activeOpacity={0.86}
              >
                <Text style={[styles.challengeTitle, { color: isSelected ? colors.primaryDark : colors.text }]}>{challenge}</Text>
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
        <Button title="Continue" size="lg" onPress={finishOnboarding} style={styles.cta} />
        <TouchableOpacity style={styles.skipButton} onPress={finishOnboarding} activeOpacity={0.8}>
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
    width: '54%',
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
    marginTop: 4,
  },
  optionalText: {
    ...Typography.captionMedium,
  },
  subtitle: {
    ...Typography.body,
    marginTop: 8,
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
  challengeCard: {
    width: '48%',
    minHeight: 104,
    borderWidth: 1,
    borderRadius: BorderRadius.lg,
    paddingHorizontal: Spacing.md,
    paddingVertical: 12,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: Spacing.md,
    ...Shadow.sm,
  },
  challengeTitle: {
    ...Typography.bodyMedium,
    flex: 1,
    fontSize: 15,
    lineHeight: 21,
  },
  checkboxOuter: {
    width: 24,
    height: 24,
    borderRadius: 12,
    borderWidth: 1.5,
    alignItems: 'center',
    justifyContent: 'center',
  },
  checkboxMark: {
    color: '#ffffff',
    fontSize: 13,
    fontWeight: '800',
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
