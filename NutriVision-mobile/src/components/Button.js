import React from 'react';
import { TouchableOpacity, Text, StyleSheet, ActivityIndicator } from 'react-native';
import { useTheme } from '../theme/ThemeContext';
import { Typography, BorderRadius, Spacing } from '../theme';

export default function Button({
  title,
  onPress,
  variant = 'primary',
  size = 'md',
  loading = false,
  disabled = false,
  icon,
  style,
}) {
  const { colors } = useTheme();

  const buttonStyles = [
    styles.base,
    size === 'lg' && styles.lg,
    size === 'sm' && styles.sm,
    {
      backgroundColor: variant === 'primary' ? colors.primary : 'transparent',
      borderColor: variant === 'outline' ? colors.border : 'transparent',
      borderWidth: variant === 'outline' ? 1.5 : 0,
      opacity: disabled ? 0.5 : 1,
    },
    style,
  ];

  const textStyles = [
    styles.text,
    size === 'lg' && styles.textLg,
    size === 'sm' && styles.textSm,
    {
      color: variant === 'primary' ? '#ffffff' : colors.primary,
    },
  ];

  return (
    <TouchableOpacity
      style={buttonStyles}
      onPress={onPress}
      disabled={disabled || loading}
      activeOpacity={0.8}
    >
      {loading ? (
        <ActivityIndicator color={variant === 'primary' ? '#fff' : colors.primary} size="small" />
      ) : (
        <>
          {icon && icon}
          <Text style={textStyles}>{title}</Text>
        </>
      )}
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  base: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 14,
    paddingHorizontal: Spacing.xxl,
    borderRadius: BorderRadius.full,
    gap: Spacing.sm,
  },
  lg: {
    paddingVertical: 16,
    paddingHorizontal: Spacing.xxxl,
  },
  sm: {
    paddingVertical: 10,
    paddingHorizontal: Spacing.lg,
  },
  text: {
    ...Typography.button,
    textAlign: 'center',
  },
  textLg: {
    fontSize: 17,
  },
  textSm: {
    fontSize: 14,
  },
});
