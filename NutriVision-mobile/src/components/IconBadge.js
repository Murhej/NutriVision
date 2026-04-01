import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { useTheme } from '../theme/ThemeContext';
import { BorderRadius, Spacing, Typography } from '../theme';

export default function IconBadge({ icon, label, color, size = 72 }) {
  const { colors } = useTheme();

  return (
    <View style={styles.container}>
      <View
        style={[
          styles.circle,
          {
            width: size,
            height: size,
            borderRadius: size / 2,
            backgroundColor: color || colors.primaryLight,
          },
        ]}
      >
        <Text style={[styles.icon, { fontSize: size * 0.4 }]}>{icon}</Text>
      </View>
      {label && (
        <Text style={[styles.label, { color: colors.text }]}>{label}</Text>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    alignItems: 'center',
    gap: Spacing.sm,
  },
  circle: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  icon: {
    textAlign: 'center',
  },
  label: {
    ...Typography.captionMedium,
    textAlign: 'center',
  },
});
