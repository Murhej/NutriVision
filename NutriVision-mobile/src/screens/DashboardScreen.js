import React from 'react';
import { View, Text, StyleSheet, ScrollView, TouchableOpacity, Image } from 'react-native';
import { useTheme } from '../theme/ThemeContext';
import { Typography, Spacing, BorderRadius, Shadow } from '../theme';
import { Card, ProgressRing, ProgressBar, MealCard } from '../components';
import { TODAY_NUTRITION, TODAYS_MEALS, USER_PROFILE } from '../data/mockData';

function getGreeting() {
  const hour = new Date().getHours();
  if (hour < 12) return 'Good Morning';
  if (hour < 17) return 'Good Afternoon';
  return 'Good Evening';
}

export default function DashboardScreen({ navigation }) {
  const { colors } = useTheme();
  const { calories, protein, carbs, fat } = TODAY_NUTRITION;

  const macros = [
    { label: 'Protein', current: protein.consumed, goal: protein.goal, unit: 'g', color: colors.protein },
    { label: 'Carbs', current: carbs.consumed, goal: carbs.goal, unit: 'g', color: colors.carbs },
    { label: 'Fat', current: fat.consumed, goal: fat.goal, unit: 'g', color: colors.fat },
  ];

  return (
    <ScrollView
      style={[styles.container, { backgroundColor: colors.background }]}
      contentContainerStyle={styles.content}
      showsVerticalScrollIndicator={false}
    >
      {/* Greeting */}
      <View style={styles.header}>
        <View>
          <Text style={[styles.greeting, { color: colors.textSecondary }]}>{getGreeting()} 👋</Text>
          <Text style={[styles.userName, { color: colors.text }]}>{USER_PROFILE.name}</Text>
        </View>
        <TouchableOpacity
          style={[styles.avatar, { backgroundColor: colors.primarySoft }]}
          onPress={() => navigation.navigate('Profile')}
        >
          {USER_PROFILE.avatar ? (
            <Image source={USER_PROFILE.avatar} style={{ width: 44, height: 44, borderRadius: 22 }} />
          ) : (
            <Text style={[styles.avatarText, { color: colors.primary }]}>{USER_PROFILE.initials}</Text>
          )}
        </TouchableOpacity>
      </View>

      {/* Calorie Ring */}
      <Card style={styles.calorieCard}>
        <View style={styles.calorieRow}>
          <ProgressRing
            size={140}
            strokeWidth={12}
            progress={calories.consumed / calories.goal}
            color={colors.calories}
            trackColor={colors.surfaceSecondary}
          >
            <Text style={[styles.calNumber, { color: colors.text }]}>{calories.consumed}</Text>
            <Text style={[styles.calLabel, { color: colors.textSecondary }]}>
              / {calories.goal} kcal
            </Text>
          </ProgressRing>

          <View style={styles.calorieInfo}>
            <Text style={[styles.calorieTitle, { color: colors.text }]}>Daily Calories</Text>
            <Text style={[styles.calorieSubtitle, { color: colors.textSecondary }]}>
              {calories.goal - calories.consumed > 0
                ? `${calories.goal - calories.consumed} kcal remaining`
                : 'Goal reached! 🎉'}
            </Text>
            <View style={[styles.streakBadge, { backgroundColor: colors.primarySoft }]}>
              <Text style={[styles.streakText, { color: colors.primary }]}>
                🔥 {USER_PROFILE.streak} day streak
              </Text>
            </View>
          </View>
        </View>
      </Card>

      {/* Macro Cards */}
      <Text style={[styles.sectionTitle, { color: colors.text }]}>Macros</Text>
      <View style={styles.macroRow}>
        {macros.map((macro, index) => (
          <Card key={index} style={styles.macroCard}>
            <Text style={[styles.macroLabel, { color: colors.textSecondary }]}>{macro.label}</Text>
            <Text style={[styles.macroValue, { color: colors.text }]}>
              {macro.current}
              <Text style={[styles.macroUnit, { color: colors.textTertiary }]}>
                /{macro.goal}{macro.unit}
              </Text>
            </Text>
            <ProgressBar
              progress={macro.current / macro.goal}
              color={macro.color}
              height={6}
              style={styles.macroBar}
            />
          </Card>
        ))}
      </View>

      {/* Quick Scan CTA */}
      <TouchableOpacity
        style={[styles.scanCta, { backgroundColor: colors.primary }]}
        onPress={() => navigation.navigate('Scan')}
        activeOpacity={0.85}
      >
        <Text style={styles.scanIcon}>📸</Text>
        <View style={styles.scanCtaText}>
          <Text style={styles.scanTitle}>Scan Your Meal</Text>
          <Text style={styles.scanSubtitle}>Take a photo to analyze nutrition</Text>
        </View>
        <Text style={styles.scanArrow}>→</Text>
      </TouchableOpacity>

      {/* Today's Meals */}
      <Text style={[styles.sectionTitle, { color: colors.text }]}>Today's Meals</Text>
      <View style={styles.mealList}>
        {TODAYS_MEALS.map((meal) => (
          <MealCard key={meal.id} meal={meal} />
        ))}
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  content: {
    padding: Spacing.lg,
    paddingBottom: Spacing.xxxl * 2,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: Spacing.xl,
  },
  greeting: {
    ...Typography.caption,
    marginBottom: 2,
  },
  userName: {
    ...Typography.h1,
  },
  avatar: {
    width: 44,
    height: 44,
    borderRadius: 22,
    alignItems: 'center',
    justifyContent: 'center',
  },
  avatarText: {
    ...Typography.bodyMedium,
    fontWeight: '700',
  },
  calorieCard: {
    marginBottom: Spacing.xl,
  },
  calorieRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.xl,
  },
  calNumber: {
    ...Typography.h1,
    fontSize: 26,
  },
  calLabel: {
    ...Typography.caption,
    marginTop: 2,
  },
  calorieInfo: {
    flex: 1,
    gap: Spacing.sm,
  },
  calorieTitle: {
    ...Typography.h3,
  },
  calorieSubtitle: {
    ...Typography.caption,
    lineHeight: 18,
  },
  streakBadge: {
    alignSelf: 'flex-start',
    paddingHorizontal: Spacing.md,
    paddingVertical: Spacing.xs,
    borderRadius: BorderRadius.full,
    marginTop: Spacing.xs,
  },
  streakText: {
    ...Typography.captionMedium,
  },
  sectionTitle: {
    ...Typography.h3,
    marginBottom: Spacing.md,
    marginTop: Spacing.sm,
  },
  macroRow: {
    flexDirection: 'row',
    gap: Spacing.md,
    marginBottom: Spacing.xl,
  },
  macroCard: {
    flex: 1,
    padding: Spacing.md,
  },
  macroLabel: {
    ...Typography.small,
    marginBottom: Spacing.xs,
  },
  macroValue: {
    ...Typography.h3,
    marginBottom: Spacing.sm,
  },
  macroUnit: {
    ...Typography.caption,
    fontWeight: '400',
  },
  macroBar: {
    marginTop: Spacing.xs,
  },
  scanCta: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: Spacing.lg,
    borderRadius: BorderRadius.xl,
    marginBottom: Spacing.xl,
    gap: Spacing.md,
  },
  scanIcon: {
    fontSize: 28,
  },
  scanCtaText: {
    flex: 1,
  },
  scanTitle: {
    ...Typography.bodyMedium,
    color: '#ffffff',
    fontWeight: '700',
  },
  scanSubtitle: {
    ...Typography.caption,
    color: 'rgba(255,255,255,0.8)',
    marginTop: 2,
  },
  scanArrow: {
    fontSize: 20,
    color: '#ffffff',
    fontWeight: '700',
  },
  mealList: {
    gap: Spacing.md,
  },
});
