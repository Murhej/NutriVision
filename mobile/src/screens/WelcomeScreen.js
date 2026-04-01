import React from 'react';
import { View, Text, StyleSheet, ScrollView, TouchableOpacity, Image } from 'react-native';
import { useTheme } from '../theme/ThemeContext';
import { Typography, Spacing, BorderRadius, Shadow } from '../theme';
import { Button } from '../components';

export default function WelcomeScreen({ navigation }) {
  const { colors } = useTheme();

  const features = [
    { icon: '🎯', label: 'Track Goals', color: colors.primarySoft },
    { icon: '🍽️', label: 'Log Meals', color: '#fef3c7' },
    { icon: '📸', label: 'AI Scanner', color: '#dbeafe' },
    { icon: '❤️', label: 'Live Healthy', color: '#fce7f3' },
  ];

  return (
    <ScrollView
      style={[styles.container, { backgroundColor: colors.background }]}
      contentContainerStyle={styles.content}
    >
      <View style={styles.logoCircle}>
        <Image 
          source={require('../../assets/icon.png')} 
          style={{ width: 80, height: 80, borderRadius: 20 }} 
          resizeMode="cover" 
        />
      </View>

      <Text style={[styles.title, { color: colors.text }]}>
        Welcome to NutriVision
      </Text>
      <Text style={[styles.subtitle, { color: colors.textSecondary }]}>
        Your AI-powered nutrition companion for a healthier lifestyle
      </Text>

      <View style={styles.grid}>
        {features.map((feature, index) => (
          <TouchableOpacity
            key={index}
            style={[
              styles.featureCard,
              {
                backgroundColor: colors.surface,
                borderColor: colors.border,
                shadowColor: colors.shadowColor,
                shadowOpacity: colors.shadowOpacity,
              },
            ]}
            activeOpacity={0.7}
          >
            <View style={[styles.featureIcon, { backgroundColor: feature.color }]}>
              <Text style={styles.featureEmoji}>{feature.icon}</Text>
            </View>
            <Text style={[styles.featureLabel, { color: colors.text }]}>
              {feature.label}
            </Text>
          </TouchableOpacity>
        ))}
      </View>

      <View style={{ width: '100%', gap: 12, marginTop: 20 }}>
        <Button
          title="Log In"
          size="lg"
          onPress={() => navigation.navigate('Login')}
          style={styles.cta}
        />
        <Button
          title="Create Account"
          size="lg"
          variant="outline"
          onPress={() => navigation.navigate('Register')}
          style={styles.cta}
        />
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  content: {
    flexGrow: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: Spacing.xxl,
    paddingVertical: Spacing.xxxl * 2,
  },
  logoCircle: {
    width: 80,
    height: 80,
    borderRadius: 40,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: Spacing.xxl,
  },
  logoEmoji: {
    fontSize: 40,
  },
  title: {
    ...Typography.hero,
    textAlign: 'center',
    marginBottom: Spacing.md,
  },
  subtitle: {
    ...Typography.body,
    textAlign: 'center',
    maxWidth: 280,
    lineHeight: 22,
    marginBottom: Spacing.xxxl,
  },
  grid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'center',
    gap: Spacing.md,
    marginBottom: Spacing.xxxl,
    width: '100%',
  },
  featureCard: {
    width: '46%',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: Spacing.xxl,
    paddingHorizontal: Spacing.lg,
    borderRadius: BorderRadius.lg,
    borderWidth: 1,
    ...Shadow.sm,
    gap: Spacing.md,
  },
  featureIcon: {
    width: 52,
    height: 52,
    borderRadius: 26,
    alignItems: 'center',
    justifyContent: 'center',
  },
  featureEmoji: {
    fontSize: 26,
  },
  featureLabel: {
    ...Typography.bodyMedium,
    textAlign: 'center',
  },
  cta: {
    width: '100%',
    maxWidth: 320,
  },
});
