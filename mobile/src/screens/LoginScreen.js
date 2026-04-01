import React, { useState, useContext } from 'react';
import { View, Text, StyleSheet, TextInput, Alert, ActivityIndicator } from 'react-native';
import { useTheme } from '../theme/ThemeContext';
import { AuthContext } from '../context/AuthContext';
import { Typography, Spacing, BorderRadius } from '../theme';
import { Button } from '../components';

export default function LoginScreen({ navigation }) {
  const { colors } = useTheme();
  const { login } = useContext(AuthContext);
  
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);

  const handleLogin = async () => {
    if (!email || !password) {
      Alert.alert('Error', 'Please fill in all fields');
      return;
    }
    
    setLoading(true);
    try {
      await login(email, password);
    } catch (error) {
      Alert.alert('Login Failed', 'Invalid email or password');
    } finally {
      setLoading(false);
    }
  };

  return (
    <View style={[styles.container, { backgroundColor: colors.background }]}>
      <Text style={[styles.title, { color: colors.text }]}>Welcome Back</Text>
      <Text style={[styles.subtitle, { color: colors.textSecondary }]}>Log in to continue</Text>

      <View style={styles.form}>
        <Text style={[styles.label, { color: colors.text }]}>Email</Text>
        <TextInput
          style={[styles.input, { borderColor: colors.border, color: colors.text, backgroundColor: colors.surface }]}
          placeholder="you@email.com"
          placeholderTextColor={colors.textTertiary}
          value={email}
          onChangeText={setEmail}
          autoCapitalize="none"
          keyboardType="email-address"
        />

        <Text style={[styles.label, { color: colors.text }]}>Password</Text>
        <TextInput
          style={[styles.input, { borderColor: colors.border, color: colors.text, backgroundColor: colors.surface }]}
          placeholder="••••••••"
          placeholderTextColor={colors.textTertiary}
          value={password}
          onChangeText={setPassword}
          secureTextEntry
        />

        <Button
          title={loading ? "Logging in..." : "Log In"}
          size="lg"
          onPress={handleLogin}
          disabled={loading}
          style={{ marginTop: Spacing.md }}
        />

        <Button
          title="Don't have an account? Sign Up"
          variant="ghost"
          onPress={() => navigation.replace('Register')}
          style={{ marginTop: Spacing.sm }}
        />
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, padding: Spacing.xxl, justifyContent: 'center' },
  title: { ...Typography.h1, marginBottom: Spacing.xs, textAlign: 'center' },
  subtitle: { ...Typography.body, marginBottom: Spacing.xxxl, textAlign: 'center' },
  form: { gap: Spacing.md },
  label: { ...Typography.captionMedium },
  input: {
    borderWidth: 1,
    borderRadius: BorderRadius.md,
    padding: Spacing.md,
    fontSize: 16,
  }
});
