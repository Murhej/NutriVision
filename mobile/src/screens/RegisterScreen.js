import React, { useState, useContext } from 'react';
import { View, Text, StyleSheet, TextInput, Alert, ScrollView } from 'react-native';
import { useTheme } from '../theme/ThemeContext';
import { AuthContext } from '../context/AuthContext';
import { Typography, Spacing, BorderRadius } from '../theme';
import { Button } from '../components';

export default function RegisterScreen({ navigation }) {
  const { colors } = useTheme();
  const { register } = useContext(AuthContext);
  
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);

  const handleRegister = async () => {
    if (!name || !email || !password) {
      Alert.alert('Error', 'Please fill in all fields');
      return;
    }
    if (password.length < 6) {
      Alert.alert('Error', 'Password must be at least 6 characters');
      return;
    }
    
    setLoading(true);
    try {
      await register(name, email, password);
    } catch (error) {
      Alert.alert('Registration Failed', 'Email might already be in use');
    } finally {
      setLoading(false);
    }
  };

  return (
    <ScrollView 
      style={[styles.container, { backgroundColor: colors.background }]}
      contentContainerStyle={{ flexGrow: 1, justifyContent: 'center' }}
    >
      <Text style={[styles.title, { color: colors.text }]}>Create Account</Text>
      <Text style={[styles.subtitle, { color: colors.textSecondary }]}>Start your healthy journey tracking limits today</Text>

      <View style={styles.form}>
        <Text style={[styles.label, { color: colors.text }]}>Full Name</Text>
        <TextInput
          style={[styles.input, { borderColor: colors.border, color: colors.text, backgroundColor: colors.surface }]}
          placeholder="Jane Doe"
          placeholderTextColor={colors.textTertiary}
          value={name}
          onChangeText={setName}
          autoCapitalize="words"
        />

        <Text style={[styles.label, { color: colors.text }]}>Email Address</Text>
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
          title={loading ? "Creating..." : "Sign Up"}
          size="lg"
          onPress={handleRegister}
          disabled={loading}
          style={{ marginTop: Spacing.md }}
        />

        <Button
          title="Already have an account? Log In"
          variant="ghost"
          onPress={() => navigation.replace('Login')}
          style={{ marginTop: Spacing.sm }}
        />
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, padding: Spacing.xxl },
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
