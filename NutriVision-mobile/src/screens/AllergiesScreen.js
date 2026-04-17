import React, { useState } from 'react';
import { KeyboardAvoidingView, Platform, ScrollView, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Button } from '../components';
import { BorderRadius, Shadow, Spacing, Typography } from '../theme';
import { useTheme } from '../theme/ThemeContext';

const H_PADDING = 18;

const ALLERGY_OPTIONS = [
  'Peanuts',
  'Tree Nuts',
  'Milk/Dairy',
  'Eggs',
  'Soy',
  'Wheat',
  'Gluten',
  'Fish',
  'Shellfish',
  'Sesame',
  'Corn',
  'Nightshade',
  'Caffeine Sensitivity',
  'Lactose Intolerance',
];

export default function AllergiesScreen({ navigation, route }) {
  const { colors } = useTheme();
  const insets = useSafeAreaInsets();
  const [selectedAllergies, setSelectedAllergies] = useState([]);

  const toggleAllergy = (allergy) => {
    setSelectedAllergies((prev) =>
      prev.includes(allergy) ? prev.filter((item) => item !== allergy) : [...prev, allergy],
    );
  };

  const handleContinue = () => {
    navigation.navigate('ActivityLevel', {
      onboardingData: {
        ...(route?.params?.onboardingData || {}),
        selectedAllergies,
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
          <Text style={[styles.title, { color: colors.text }]}>Any allergies or intolerances?</Text>
          <View style={[styles.optionalBadge, { backgroundColor: colors.primarySoft, borderColor: colors.primaryLight }]}>
            <Text style={[styles.optionalText, { color: colors.primaryDark }]}>Optional</Text>
          </View>
        </View>

        <Text style={[styles.subtitle, { color: colors.textSecondary }]}>Select all that apply.</Text>

        <View style={styles.chipsWrap}>
          {ALLERGY_OPTIONS.map((option) => {
            const isSelected = selectedAllergies.includes(option);
            return (
              <TouchableOpacity
                key={option}
                style={[
                  styles.chip,
                  {
                    borderColor: isSelected ? colors.primary : colors.border,
                    backgroundColor: isSelected ? colors.primarySoft : colors.surfaceSecondary,
                  },
                ]}
                onPress={() => toggleAllergy(option)}
                activeOpacity={0.85}
              >
                <Text style={[styles.chipText, { color: isSelected ? colors.primaryDark : colors.text }]}>{option}</Text>
                {isSelected ? <Text style={[styles.chipCheck, { color: colors.primary }]}>✓</Text> : null}
              </TouchableOpacity>
            );
          })}
        </View>

        <Text style={[styles.helperText, { color: colors.textTertiary }]}>No allergies? You can continue to the next step.</Text>
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
    width: '86%',
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
    marginBottom: 14,
  },
  chipsWrap: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
  },
  chip: {
    borderWidth: 1,
    borderRadius: BorderRadius.full,
    paddingVertical: 12,
    paddingHorizontal: 18,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    minHeight: 46,
  },
  chipText: {
    ...Typography.bodyMedium,
    fontSize: 15,
  },
  chipCheck: {
    ...Typography.captionMedium,
    fontSize: 14,
  },
  helperText: {
    ...Typography.body,
    fontSize: 14,
    marginTop: 14,
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
