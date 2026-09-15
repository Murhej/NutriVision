import React, { useMemo, useState } from 'react';
import {
  Image,
  KeyboardAvoidingView,
  Modal,
  Platform,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Button } from '../components';
import { BorderRadius, Shadow, Spacing, Typography } from '../theme';
import { useTheme } from '../theme/ThemeContext';
import { useAuth } from '../context/AuthContext';

const H_PADDING = 18;

function joinList(values, fallback = 'Not specified') {
  if (!Array.isArray(values) || values.length === 0) return fallback;
  return values.join(', ');
}

function isValidEmail(value) {
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value);
}

function isStrongPassword(value) {
  return /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[^A-Za-z0-9]).{8,}$/.test(value);
}

function inferGoalType(goalText = '') {
  const lower = String(goalText || '').toLowerCase();
  if (lower.includes('muscle') || lower.includes('lean') || lower.includes('bulk')) return 'muscle';
  if (lower.includes('lose') || lower.includes('fat')) return 'lose';
  if (lower.includes('gain') || lower.includes('build')) return 'gain';
  return 'maintain';
}

function SummaryRow({ label, value, colors }) {
  return (
    <View style={[styles.row, { borderBottomColor: colors.borderLight }]}>
      <Text style={[styles.rowLabel, { color: colors.textSecondary }]}>{label}</Text>
      <Text style={[styles.rowValue, { color: colors.text }]}>{value}</Text>
    </View>
  );
}

export default function ProfileReviewScreen({ navigation, route }) {
  const { colors } = useTheme();
  const { register, updateProfile, uploadAvatar } = useAuth();
  const insets = useSafeAreaInsets();
  const [isSignUpOpen, setIsSignUpOpen] = useState(false);
  const [profileImageUri, setProfileImageUri] = useState('');
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [touched, setTouched] = useState({});
  const [formMessage, setFormMessage] = useState('');

  const onboardingData = route?.params?.onboardingData || {};
  const personalInfo = onboardingData.personalInfo || {};

  const errors = useMemo(() => {
    const fieldErrors = {};

    if (!profileImageUri) {
      fieldErrors.profileImage = 'Please upload a profile picture.';
    }

    if (!username.trim()) {
      fieldErrors.username = 'Please enter a username.';
    } else if (username.trim().length < 3) {
      fieldErrors.username = 'Username must be at least 3 characters.';
    }

    if (!email.trim() || !isValidEmail(email.trim())) {
      fieldErrors.email = 'Enter a valid email address.';
    }

    if (!password || !isStrongPassword(password)) {
      fieldErrors.password = 'Password must be at least 8 characters and include uppercase, lowercase, number, and symbol.';
    }

    if (!confirmPassword || confirmPassword !== password) {
      fieldErrors.confirmPassword = 'Passwords do not match.';
    }

    return fieldErrors;
  }, [confirmPassword, email, password, profileImageUri, username]);

  const canSignUp = Object.keys(errors).length === 0;

  const summaryPersonalRows = [
    { label: 'Name', value: personalInfo.fullName || [personalInfo.firstName, personalInfo.lastName].filter(Boolean).join(' ') || 'Not specified' },
    { label: 'Age', value: personalInfo.age || 'Not specified' },
    { label: 'Gender', value: personalInfo.gender || 'Not specified' },
    { label: 'Height', value: personalInfo.height || 'Not specified' },
    { label: 'Weight', value: personalInfo.weight || 'Not specified' },
    ...(personalInfo.targetWeight ? [{ label: 'Target Weight', value: personalInfo.targetWeight }] : []),
    { label: 'Country', value: personalInfo.country || 'Not specified' },
  ];

  const summaryPreferenceRows = [
    { label: 'Main Goal', value: onboardingData.customGoal || (onboardingData.selectedGoals && onboardingData.selectedGoals[0]) || 'Not specified' },
    { label: 'Challenges', value: joinList(onboardingData.selectedChallenges, 'None listed') },
    { label: 'Dietary Preferences', value: joinList(onboardingData.selectedDietaryPreferences, 'None listed') },
    { label: 'Allergies', value: joinList(onboardingData.selectedAllergies, 'None listed') },
    { label: 'Activity', value: onboardingData.activity || 'Not specified' },
    { label: 'Exercise Level', value: onboardingData.activityLevel || 'Not specified' },
    { label: 'Weekly Exercise', value: onboardingData.weeklyExerciseDuration || 'Not specified' },
    { label: 'Work Type', value: onboardingData.workType || 'Not specified' },
    { label: 'Training Setup', value: joinList(onboardingData.trainingEquipment, 'Not specified') },
    { label: 'Training Focus', value: joinList(onboardingData.trainingStyles, 'Not specified') },
  ];

  const markAllTouched = () => {
    setTouched({
      profileImage: true,
      username: true,
      email: true,
      password: true,
      confirmPassword: true,
    });
  };

  const pickProfileImage = async () => {
    const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!permission.granted) {
      setFormMessage('Please allow photo library access to upload a profile picture.');
      return;
    }

    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      quality: 0.85,
      aspect: [1, 1],
    });

    if (!result.canceled && result.assets && result.assets[0]?.uri) {
      setProfileImageUri(result.assets[0].uri);
      setTouched((prev) => ({ ...prev, profileImage: true }));
      setFormMessage('');
    }
  };

  const handleSignUp = async () => {
    markAllTouched();
    if (!canSignUp) return;

    try {
      setFormMessage('Creating account...');

      const registerResult = await register({
        name: personalInfo.fullName || [personalInfo.firstName, personalInfo.lastName].filter(Boolean).join(' ') || username.trim(),
        email: email.trim(),
        password,
      });

      await updateProfile({
        username: username.trim(),
        fullName: personalInfo.fullName || [personalInfo.firstName, personalInfo.lastName].filter(Boolean).join(' ') || undefined,
        country: personalInfo.country || '',
        goal: onboardingData.customGoal || (onboardingData.selectedGoals && onboardingData.selectedGoals[0]) || '',
        goalType: inferGoalType(onboardingData.customGoal || (onboardingData.selectedGoals && onboardingData.selectedGoals[0]) || ''),
        dietaryPreferences: onboardingData.selectedDietaryPreferences || [],
        allergies: onboardingData.selectedAllergies || [],
        activityLevel: onboardingData.activityLevel || onboardingData.activity || '',
        exerciseHabits: onboardingData.weeklyExerciseDuration || '',
      }, registerResult?.token || null);

      if (profileImageUri) {
        await uploadAvatar(profileImageUri, registerResult?.token || null);
      }

      setFormMessage('');
      setIsSignUpOpen(false);
    } catch (error) {
      setFormMessage(error?.message || 'Unable to complete sign up.');
    }
  };

  const showFieldError = (field) => touched[field] && errors[field];

  return (
    <KeyboardAvoidingView
      style={[styles.container, { backgroundColor: colors.background }]}
      behavior={Platform.OS === 'ios' ? 'padding' : undefined}
    >
      <View
        style={[
          styles.stickyTop,
          {
            backgroundColor: colors.background,
            paddingTop: insets.top + 8,
            borderBottomColor: colors.borderLight,
          },
        ]}
      >
        <TouchableOpacity
          style={[styles.backButton, { backgroundColor: colors.surface, borderColor: colors.border }]}
          onPress={() => navigation.goBack()}
          activeOpacity={0.8}
        >
          <Text style={[styles.backText, { color: colors.textSecondary }]}>{'<'}</Text>
        </TouchableOpacity>

        <View style={[styles.progressTrack, { backgroundColor: colors.borderLight }]}>
          <View style={[styles.progressFill, { backgroundColor: colors.primary }]} />
        </View>
      </View>

      <ScrollView
        style={styles.scrollView}
        contentContainerStyle={[styles.content, { paddingBottom: 170 + insets.bottom }]}
        showsVerticalScrollIndicator={false}
      >
        <View style={styles.headerRow}>
          <Text style={[styles.title, { color: colors.text }]}>Your Profile Is Ready</Text>
          <View style={[styles.finalBadge, { backgroundColor: colors.primarySoft, borderColor: colors.primaryLight }]}>
            <Text style={[styles.finalBadgeText, { color: colors.primaryDark }]}>Final Step</Text>
          </View>
        </View>
        <Text style={[styles.subtitle, { color: colors.textSecondary }]}>Review your setup, then create your account to start tracking.</Text>

        <View
          style={[
            styles.statusCard,
            {
              backgroundColor: colors.primarySoft,
              borderColor: colors.primaryLight,
            },
          ]}
        >
          <View style={[styles.statusIcon, { borderColor: colors.primaryLight, backgroundColor: colors.surface }]}>
            <Text style={[styles.statusIconText, { color: colors.primary }]}>*</Text>
          </View>
          <View style={styles.statusTextWrap}>
            <Text style={[styles.statusTitle, { color: colors.text }]}>Profile summary complete</Text>
            <Text style={[styles.statusSubtitle, { color: colors.textSecondary }]}>You can still edit goals and settings later from Home and Profile.</Text>
          </View>
        </View>

        <View style={[styles.sectionCard, { backgroundColor: colors.surface, borderColor: colors.border, shadowColor: colors.shadowColor, shadowOpacity: colors.shadowOpacity }]}>
          <Text style={[styles.sectionTitle, { color: colors.text }]}>Personal Info</Text>
          {summaryPersonalRows.map((row) => (
            <SummaryRow key={row.label} label={row.label} value={row.value} colors={colors} />
          ))}
        </View>

        <View style={[styles.sectionCard, { backgroundColor: colors.surface, borderColor: colors.border, shadowColor: colors.shadowColor, shadowOpacity: colors.shadowOpacity }]}>
          <Text style={[styles.sectionTitle, { color: colors.text }]}>Goals and Preferences</Text>
          {summaryPreferenceRows.map((row) => (
            <SummaryRow key={row.label} label={row.label} value={row.value} colors={colors} />
          ))}
        </View>
      </ScrollView>

      <View
        style={[
          styles.stickyBottom,
          {
            paddingBottom: insets.bottom + Spacing.md,
            backgroundColor: colors.background,
            borderTopColor: colors.borderLight,
          },
        ]}
      >
        <Button title="Continue" size="lg" onPress={() => setIsSignUpOpen(true)} style={styles.cta} />
      </View>

      <Modal visible={isSignUpOpen} transparent animationType="fade" onRequestClose={() => setIsSignUpOpen(false)}>
        <View style={styles.modalOverlay}>
          <View style={[styles.modalCard, { backgroundColor: colors.surface }]}> 
            <ScrollView contentContainerStyle={styles.modalContent} showsVerticalScrollIndicator={false} keyboardShouldPersistTaps="handled">
              <Text style={[styles.modalTitle, { color: colors.text }]}>Continue Your Journey</Text>
              <Text style={[styles.modalSubtitle, { color: colors.textSecondary }]}>Create your account to save your profile and start your plan.</Text>

              <TouchableOpacity
                style={[styles.uploadButton, { borderColor: colors.primaryLight, backgroundColor: colors.primarySoft }]}
                onPress={pickProfileImage}
                activeOpacity={0.85}
              >
                {profileImageUri ? <Image source={{ uri: profileImageUri }} style={styles.profilePreview} /> : null}
                <Text style={[styles.uploadText, { color: colors.primaryDark }]}>{profileImageUri ? 'Change Profile Picture' : 'Upload Profile Picture'}</Text>
              </TouchableOpacity>
              {showFieldError('profileImage') ? <Text style={[styles.fieldError, { color: colors.danger }]}>{errors.profileImage}</Text> : null}

              <Text style={[styles.fieldLabel, { color: colors.textSecondary }]}>Username</Text>
              <TextInput
                style={[styles.fieldInput, { borderColor: showFieldError('username') ? colors.danger : colors.border, color: colors.text, backgroundColor: colors.surfaceSecondary }]}
                placeholder="Enter username"
                placeholderTextColor={colors.textTertiary}
                value={username}
                onChangeText={setUsername}
                onBlur={() => setTouched((prev) => ({ ...prev, username: true }))}
                autoCapitalize="none"
              />
              {showFieldError('username') ? <Text style={[styles.fieldError, { color: colors.danger }]}>{errors.username}</Text> : null}

              <Text style={[styles.fieldLabel, { color: colors.textSecondary }]}>Email</Text>
              <TextInput
                style={[styles.fieldInput, { borderColor: showFieldError('email') ? colors.danger : colors.border, color: colors.text, backgroundColor: colors.surfaceSecondary }]}
                placeholder="Enter your email"
                placeholderTextColor={colors.textTertiary}
                value={email}
                onChangeText={setEmail}
                onBlur={() => setTouched((prev) => ({ ...prev, email: true }))}
                autoCapitalize="none"
                keyboardType="email-address"
              />
              {showFieldError('email') ? <Text style={[styles.fieldError, { color: colors.danger }]}>{errors.email}</Text> : null}

              <Text style={[styles.fieldLabel, { color: colors.textSecondary }]}>Password</Text>
              <TextInput
                style={[styles.fieldInput, { borderColor: showFieldError('password') ? colors.danger : colors.border, color: colors.text, backgroundColor: colors.surfaceSecondary }]}
                placeholder="Enter password"
                placeholderTextColor={colors.textTertiary}
                value={password}
                onChangeText={setPassword}
                onBlur={() => setTouched((prev) => ({ ...prev, password: true }))}
                secureTextEntry
              />
              {showFieldError('password') ? <Text style={[styles.fieldError, { color: colors.danger }]}>{errors.password}</Text> : null}

              <Text style={[styles.fieldLabel, { color: colors.textSecondary }]}>Confirm Password</Text>
              <TextInput
                style={[styles.fieldInput, { borderColor: showFieldError('confirmPassword') ? colors.danger : colors.border, color: colors.text, backgroundColor: colors.surfaceSecondary }]}
                placeholder="Confirm password"
                placeholderTextColor={colors.textTertiary}
                value={confirmPassword}
                onChangeText={setConfirmPassword}
                onBlur={() => setTouched((prev) => ({ ...prev, confirmPassword: true }))}
                secureTextEntry
              />
              {showFieldError('confirmPassword') ? <Text style={[styles.fieldError, { color: colors.danger }]}>{errors.confirmPassword}</Text> : null}

              {formMessage ? <Text style={[styles.formMessage, { color: colors.warning }]}>{formMessage}</Text> : null}

              <TouchableOpacity style={styles.loginLink} onPress={() => navigation.navigate('Login')} activeOpacity={0.8}>
                <Text style={[styles.loginLinkText, { color: colors.primary }]}>Already have an account? Log in</Text>
              </TouchableOpacity>

              <View style={styles.modalActions}>
                <Button title="Maybe Later" size="sm" variant="outline" onPress={() => setIsSignUpOpen(false)} style={styles.secondaryAction} />
                <Button title="Sign Up" size="sm" onPress={handleSignUp} style={[styles.primaryAction, !canSignUp && styles.blockedPrimary]} />
              </View>
            </ScrollView>
          </View>
        </View>
      </Modal>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  stickyTop: {
    paddingHorizontal: H_PADDING,
    paddingBottom: 10,
    borderBottomWidth: 1,
  },
  backButton: {
    width: 44,
    height: 44,
    borderRadius: 22,
    borderWidth: 1,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 10,
    ...Shadow.sm,
  },
  backText: {
    ...Typography.h2,
    lineHeight: 20,
  },
  progressTrack: {
    height: 6,
    borderRadius: BorderRadius.full,
    overflow: 'hidden',
  },
  progressFill: {
    width: '100%',
    height: '100%',
    borderRadius: BorderRadius.full,
  },
  scrollView: {
    flex: 1,
  },
  content: {
    paddingHorizontal: H_PADDING,
    paddingTop: 18,
    gap: 12,
  },
  headerRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    gap: Spacing.md,
  },
  title: {
    ...Typography.hero,
    fontSize: 24,
    flex: 1,
  },
  finalBadge: {
    borderWidth: 1,
    borderRadius: BorderRadius.full,
    paddingHorizontal: Spacing.md,
    paddingVertical: 6,
    marginTop: 3,
  },
  finalBadgeText: {
    ...Typography.captionMedium,
  },
  subtitle: {
    ...Typography.body,
    fontSize: 15,
    lineHeight: 24,
  },
  statusCard: {
    borderWidth: 1,
    borderRadius: BorderRadius.lg,
    padding: 14,
    flexDirection: 'row',
    gap: Spacing.md,
    alignItems: 'center',
  },
  statusIcon: {
    width: 52,
    height: 52,
    borderRadius: 26,
    borderWidth: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  statusIconText: {
    ...Typography.h2,
  },
  statusTextWrap: {
    flex: 1,
  },
  statusTitle: {
    ...Typography.h3,
    marginBottom: 2,
  },
  statusSubtitle: {
    ...Typography.body,
    lineHeight: 24,
  },
  sectionCard: {
    borderWidth: 1,
    borderRadius: BorderRadius.lg,
    padding: 12,
    ...Shadow.md,
  },
  sectionTitle: {
    ...Typography.hero,
    fontSize: 20,
    marginBottom: 8,
  },
  row: {
    minHeight: 52,
    paddingVertical: 8,
    borderBottomWidth: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: Spacing.md,
  },
  rowLabel: {
    ...Typography.h3,
    fontSize: 17,
    flex: 0.45,
  },
  rowValue: {
    ...Typography.h3,
    fontSize: 17,
    flex: 0.55,
    textAlign: 'right',
  },
  stickyBottom: {
    borderTopWidth: 1,
    paddingHorizontal: H_PADDING,
    paddingTop: 10,
  },
  cta: {
    width: '100%',
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(15, 23, 42, 0.7)',
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 18,
    paddingVertical: 26,
  },
  modalCard: {
    width: '100%',
    maxHeight: '92%',
    borderRadius: BorderRadius.xl,
    ...Shadow.lg,
  },
  modalContent: {
    padding: 16,
  },
  modalTitle: {
    ...Typography.h3,
    fontSize: 20,
    marginBottom: 2,
  },
  modalSubtitle: {
    ...Typography.caption,
    marginBottom: 12,
  },
  uploadButton: {
    borderWidth: 1,
    borderRadius: BorderRadius.full,
    minHeight: 48,
    alignItems: 'center',
    justifyContent: 'center',
    flexDirection: 'row',
    gap: 8,
    marginBottom: 6,
  },
  uploadText: {
    ...Typography.captionMedium,
  },
  profilePreview: {
    width: 28,
    height: 28,
    borderRadius: 14,
  },
  fieldLabel: {
    ...Typography.caption,
    marginTop: 10,
    marginBottom: 4,
  },
  fieldInput: {
    borderWidth: 1,
    borderRadius: BorderRadius.md,
    minHeight: 44,
    paddingHorizontal: 10,
    ...Typography.caption,
    fontSize: 14,
  },
  fieldError: {
    ...Typography.caption,
    marginTop: 4,
  },
  formMessage: {
    ...Typography.caption,
    marginTop: 8,
  },
  loginLink: {
    alignSelf: 'center',
    marginTop: 10,
  },
  loginLinkText: {
    ...Typography.captionMedium,
  },
  modalActions: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    gap: 10,
    marginTop: 14,
  },
  secondaryAction: {
    flex: 1,
  },
  primaryAction: {
    flex: 1,
  },
  blockedPrimary: {
    opacity: 0.55,
  },
});
