import React, { useState } from 'react';
import { View, Text, StyleSheet, TextInput, TouchableOpacity, ScrollView } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useTheme } from '../theme/ThemeContext';
import { BorderRadius, Spacing, Typography } from '../theme';
import { Card, Button } from '../components';
import { useAuth } from '../context/AuthContext';

function isValidEmail(value) {
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value || '');
}

export default function LoginScreen({ navigation }) {
  const { colors } = useTheme();
  const insets = useSafeAreaInsets();
  const { login } = useAuth();

  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleLogin = async () => {
    const emailValue = email.trim();
    if (!isValidEmail(emailValue)) {
      setError('Enter a valid email address.');
      return;
    }
    if (!password) {
      setError('Password is required.');
      return;
    }

    try {
      setLoading(true);
      setError('');
      await login({ email: emailValue, password });
    } catch (e) {
      setError(e?.message || 'Unable to sign in.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <ScrollView
      style={[styles.container, { backgroundColor: colors.background }]}
      contentContainerStyle={[styles.content, { paddingBottom: insets.bottom + 24 }]}
      keyboardShouldPersistTaps="handled"
    >
      <TouchableOpacity
        style={[styles.backButton, { borderColor: colors.border, backgroundColor: colors.surface }]}
        activeOpacity={0.8}
        onPress={() => navigation.goBack()}
      >
        <Text style={[styles.backText, { color: colors.textSecondary }]}>{'<'}</Text>
      </TouchableOpacity>

      <Text style={[styles.title, { color: colors.text }]}>Sign In</Text>
      <Text style={[styles.subtitle, { color: colors.textSecondary }]}>Use your NutriVision account to continue.</Text>

      <Card style={styles.formCard}>
        <Text style={[styles.label, { color: colors.textSecondary }]}>Email</Text>
        <TextInput
          style={[styles.input, { borderColor: colors.border, backgroundColor: colors.surfaceSecondary, color: colors.text }]}
          value={email}
          onChangeText={setEmail}
          autoCapitalize="none"
          keyboardType="email-address"
          placeholder="name@email.com"
          placeholderTextColor={colors.textTertiary}
        />

        <Text style={[styles.label, { color: colors.textSecondary }]}>Password</Text>
        <TextInput
          style={[styles.input, { borderColor: colors.border, backgroundColor: colors.surfaceSecondary, color: colors.text }]}
          value={password}
          onChangeText={setPassword}
          secureTextEntry
          placeholder="Password"
          placeholderTextColor={colors.textTertiary}
        />

        {error ? <Text style={[styles.error, { color: colors.danger }]}>{error}</Text> : null}

        <Button title={loading ? 'Signing In...' : 'Sign In'} onPress={handleLogin} disabled={loading} size="lg" />

        <TouchableOpacity style={styles.signUpHint} onPress={() => navigation.navigate('PersonalDetails')} activeOpacity={0.8}>
          <Text style={[styles.signUpHintText, { color: colors.primary }]}>Need an account? Create one</Text>
        </TouchableOpacity>
      </Card>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  content: {
    padding: Spacing.lg,
    gap: Spacing.md,
  },
  backButton: {
    width: 40,
    height: 40,
    borderRadius: BorderRadius.full,
    borderWidth: 1,
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: Spacing.md,
  },
  backText: {
    ...Typography.bodyMedium,
    fontWeight: '700',
  },
  title: {
    ...Typography.h1,
    marginTop: Spacing.md,
  },
  subtitle: {
    ...Typography.body,
    marginBottom: Spacing.md,
  },
  formCard: {
    gap: Spacing.sm,
  },
  label: {
    ...Typography.captionMedium,
    marginTop: Spacing.xs,
  },
  input: {
    borderWidth: 1,
    borderRadius: BorderRadius.md,
    paddingHorizontal: Spacing.md,
    paddingVertical: Spacing.md,
    ...Typography.body,
  },
  error: {
    ...Typography.caption,
    marginTop: 2,
  },
  signUpHint: {
    alignItems: 'center',
    marginTop: Spacing.sm,
  },
  signUpHintText: {
    ...Typography.captionMedium,
  },
});
