import React from 'react';
import { View, Text, StyleSheet, ScrollView, Image } from 'react-native';
import { useTheme } from '../theme/ThemeContext';
import { Typography, Spacing } from '../theme';
import { Button } from '../components';

export default function WelcomeScreen({ navigation }) {
  const { colors } = useTheme();

  return (
    <ScrollView
      style={[styles.container, { backgroundColor: colors.background }]}
      contentContainerStyle={styles.content}
    >
      <View style={styles.logoCircle}>
        <Image
          source={require('../../assets/icon.png')}
          style={styles.logoImage}
          resizeMode="cover"
        />
      </View>

      <Text style={[styles.title, { color: colors.text }]}>
        Welcome to NutriVision
      </Text>

      <Text style={[styles.subtitle, { color: colors.textSecondary }]}>
        Your AI-powered nutrition companion for a healthier lifestyle
      </Text>

      <Button
        title="Get Started"
        size="lg"
        onPress={() => navigation.navigate('PersonalDetails')}
        style={styles.primaryButton}
      />

      <Button
        title="I Already Have an Account"
        size="lg"
        onPress={() => navigation.navigate('Login')}
        style={styles.secondaryButton}
      />
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
    width: 100,
    height: 100,
    borderRadius: 50,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: Spacing.xxl,
  },
  logoImage: {
    width: 90,
    height: 90,
    borderRadius: 20,
  },
  title: {
    ...Typography.hero,
    textAlign: 'center',
    marginBottom: Spacing.md,
  },
  subtitle: {
    ...Typography.body,
    textAlign: 'center',
    maxWidth: 300,
    lineHeight: 22,
    marginBottom: Spacing.xxxl,
  },
  primaryButton: {
    width: '100%',
    maxWidth: 320,
    marginBottom: Spacing.lg,
  },
  secondaryButton: {
    width: '100%',
    maxWidth: 320,
  },
});
