import React from 'react';
import { View, Text, StyleSheet, ScrollView, Image } from 'react-native';
import { useTheme } from '../theme/ThemeContext';
import { Typography, Spacing, BorderRadius, Shadow } from '../theme';
import { Button } from '../components';

export default function WelcomeScreen({ navigation }) {
  const { colors } = useTheme();

  return (
    <ScrollView
      style={[styles.container, { backgroundColor: colors.background }]}
      contentContainerStyle={styles.content}
      showsVerticalScrollIndicator={false}
    >
      <View style={styles.heroWrap}>
        <View style={[styles.glowRing, { backgroundColor: colors.primarySoft }]} />

        <View style={[styles.logoShell, { backgroundColor: colors.surface, shadowColor: colors.shadowColor, shadowOpacity: colors.shadowOpacity }]}> 
          <Image
            source={require('../../assets/icon.png')}
            style={styles.logoImage}
            resizeMode="cover"
          />
        </View>

        <View style={[styles.badge, { backgroundColor: colors.primarySoft, borderColor: colors.primaryLight }]}> 
          <Text style={[styles.badgeText, { color: colors.primaryDark }]}>AI Nutrition Coach</Text>
        </View>
      </View>

      <Text style={[styles.title, { color: colors.text }]}>
        Welcome to NutriVision
      </Text>
      <Text style={[styles.subtitle, { color: colors.textSecondary }]}>
        Personalized tracking, smart meal scans, and clear progress in one place.
      </Text>

      <View
        style={[
          styles.actionCard,
          {
            backgroundColor: colors.surface,
            borderColor: colors.border,
            shadowColor: colors.shadowColor,
            shadowOpacity: colors.shadowOpacity,
          },
        ]}
      >
        <Button
          title="Get Started"
          size="lg"
          onPress={() => navigation.navigate('PersonalDetails')}
          style={styles.cta}
        />

        <Button
          title="I Already Have an Account"
          size="lg"
          variant="outline"
          onPress={() => navigation.navigate('Login')}
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
    justifyContent: 'space-between',
    paddingHorizontal: Spacing.xxl,
    paddingTop: Spacing.xxxl,
    paddingBottom: Spacing.xxl,
  },
  heroWrap: {
    width: '100%',
    alignItems: 'center',
    justifyContent: 'center',
    paddingTop: Spacing.xl,
    minHeight: 280,
  },
  glowRing: {
    position: 'absolute',
    width: 230,
    height: 230,
    borderRadius: 115,
    opacity: 0.75,
  },
  logoShell: {
    width: 136,
    height: 136,
    borderRadius: 40,
    alignItems: 'center',
    justifyContent: 'center',
    ...Shadow.lg,
  },
  logoImage: {
    width: 104,
    height: 104,
    borderRadius: 28,
  },
  badge: {
    marginTop: Spacing.lg,
    borderWidth: 1,
    borderRadius: BorderRadius.full,
    paddingHorizontal: Spacing.lg,
    paddingVertical: Spacing.sm,
  },
  badgeText: {
    ...Typography.bodyMedium,
    fontSize: 16,
  },
  title: {
    ...Typography.hero,
    textAlign: 'center',
    fontSize: 30,
    marginTop: Spacing.xxl,
    marginBottom: Spacing.md,
  },
  subtitle: {
    ...Typography.body,
    textAlign: 'center',
    maxWidth: 320,
    fontSize: 17,
    lineHeight: 30,
  },
  actionCard: {
    width: '100%',
    borderWidth: 1,
    borderRadius: 28,
    padding: Spacing.xl,
    gap: Spacing.lg,
    ...Shadow.md,
    marginTop: Spacing.xxxl,
  },
  cta: {
    width: '100%',
    minHeight: 62,
  },
});
