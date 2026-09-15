import React from 'react';
import { View, Text, StyleSheet, Image } from 'react-native';
import { useTheme } from '../theme/ThemeContext';
import Card from './Card';
import { Typography, Spacing } from '../theme';

export default function MealCard({ meal }) {
  const { colors } = useTheme();

  return (
    <Card style={styles.card}>
      <View style={styles.row}>
        <View style={[styles.emojiCircle, { backgroundColor: colors.primarySoft, overflow: 'hidden' }]}>
          {meal.image ? (
            <Image source={meal.image} style={{ width: 44, height: 44, resizeMode: 'cover' }} />
          ) : (
            <Text style={styles.emoji}>{meal.emoji || '🍽️'}</Text>
          )}
        </View>
        <View style={styles.info}>
          <Text style={[styles.name, { color: colors.text }]} numberOfLines={1}>
            {meal.name}
          </Text>
          <Text style={[styles.meta, { color: colors.textSecondary }]}>
            {meal.type} • {meal.time}
          </Text>
        </View>
        <View style={styles.calCol}>
          <Text style={[styles.cal, { color: colors.text }]}>{meal.calories}</Text>
          <Text style={[styles.calLabel, { color: colors.textTertiary }]}>kcal</Text>
        </View>
      </View>
    </Card>
  );
}

const styles = StyleSheet.create({
  card: {
    padding: Spacing.md,
  },
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.md,
  },
  emojiCircle: {
    width: 44,
    height: 44,
    borderRadius: 22,
    alignItems: 'center',
    justifyContent: 'center',
  },
  emoji: {
    fontSize: 22,
  },
  info: {
    flex: 1,
    gap: 2,
  },
  name: {
    ...Typography.bodyMedium,
  },
  meta: {
    ...Typography.caption,
  },
  calCol: {
    alignItems: 'flex-end',
  },
  cal: {
    ...Typography.h3,
  },
  calLabel: {
    ...Typography.small,
    textTransform: 'lowercase',
    fontSize: 11,
  },
});
