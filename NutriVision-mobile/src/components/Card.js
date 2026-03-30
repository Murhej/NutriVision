import React from 'react';
import { View, StyleSheet } from 'react-native';
import { useTheme } from '../theme/ThemeContext';
import { BorderRadius, Shadow, Spacing } from '../theme';

export default function Card({ children, style, variant = 'elevated' }) {
  const { colors } = useTheme();

  return (
    <View
      style={[
        styles.base,
        {
          backgroundColor: colors.surface,
          borderColor: colors.border,
          shadowColor: colors.shadowColor,
          shadowOpacity: variant === 'elevated' ? colors.shadowOpacity : 0,
        },
        variant === 'flat' && { borderWidth: 1 },
        style,
      ]}
    >
      {children}
    </View>
  );
}

const styles = StyleSheet.create({
  base: {
    borderRadius: BorderRadius.lg,
    padding: Spacing.lg,
    ...Shadow.md,
  },
});
