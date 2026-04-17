import React, { useMemo, useState } from 'react';
import { KeyboardAvoidingView, Platform, ScrollView, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Button } from '../components';
import { BorderRadius, Shadow, Spacing, Typography } from '../theme';
import { useTheme } from '../theme/ThemeContext';

const H_PADDING = 18;

const DIETARY_OPTIONS = [
  'No Preference',
  'Balanced',
  'High Protein',
  'Low Carb',
  'Keto',
  'Paleo',
  'Mediterranean',
  'Vegetarian',
  'Vegan',
  'Pescatarian',
  'Halal',
  'Kosher',
  'Gluten-Free',
  'Dairy-Free',
  'Low Sodium',
  'Low Sugar',
  'Plant-Forward',
  'Intermittent Fasting',
];

export default function DietaryPreferencesScreen({ navigation, route }) {
  const { colors } = useTheme();
  const insets = useSafeAreaInsets();
  const [selectedPreferences, setSelectedPreferences] = useState([]);
  const [hasTriedContinue, setHasTriedContinue] = useState(false);

  const canContinue = useMemo(() => selectedPreferences.length > 0, [selectedPreferences]);

  const togglePreference = (option) => {
    setSelectedPreferences((prev) =>
      prev.includes(option) ? prev.filter((item) => item !== option) : [...prev, option],
    );
  };

  const onContinue = () => {
    setHasTriedContinue(true);
    if (!canContinue) return;

    navigation.navigate('Allergies', {
      onboardingData: {
        ...(route?.params?.onboardingData || {}),
        selectedDietaryPreferences: selectedPreferences,
      },
    });
  };

  const errorMessage = hasTriedContinue && !canContinue ? 'Please select at least one dietary preference.' : '';

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
        contentContainerStyle={[styles.content, { paddingBottom: 148 + insets.bottom }]}
        showsVerticalScrollIndicator={false}
      >
        <Text style={[styles.title, { color: colors.text }]}>Dietary Preferences</Text>
        <Text style={[styles.subtitle, { color: colors.textSecondary }]}>Select all your preferred eating styles.</Text>

        <View style={styles.grid}>
          {DIETARY_OPTIONS.map((option) => {
            const isSelected = selectedPreferences.includes(option);
            return (
              <TouchableOpacity
                key={option}
                style={[
                  styles.optionCard,
                  {
                    borderColor: isSelected ? colors.primary : errorMessage ? colors.danger : colors.border,
                    backgroundColor: isSelected ? colors.primarySoft : colors.surface,
                    shadowColor: colors.shadowColor,
                    shadowOpacity: colors.shadowOpacity,
                  },
                ]}
                onPress={() => togglePreference(option)}
                activeOpacity={0.88}
              >
                <Text style={[styles.optionText, { color: isSelected ? colors.primaryDark : colors.text }]}>{option}</Text>
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

        {errorMessage ? <Text style={[styles.errorText, { color: colors.danger }]}>{errorMessage}</Text> : null}
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
        <Button title="Continue" size="lg" onPress={onContinue} disabled={!canContinue} style={styles.cta} />
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
    width: '72%',
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
    marginBottom: 14,
  },
  grid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: Spacing.sm,
  },
  optionCard: {
    width: '48%',
    minHeight: 88,
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
  optionText: {
    ...Typography.bodyMedium,
    fontSize: 15,
    flex: 1,
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
