import React, { useMemo, useState } from 'react';
import {
  Alert,
  Image,
  Modal,
  RefreshControl,
  ScrollView,
  StyleSheet,
  Switch,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { useFocusEffect, useNavigation } from '@react-navigation/native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useTheme } from '../theme/ThemeContext';
import { Typography, Spacing, BorderRadius } from '../theme';
import { Card, Button } from '../components';
import { Bell, LogOut, Moon, Ruler, Trash2, UserRound } from 'lucide-react-native';
import { API_BASE_URL } from '../api/client';
import { useAuth } from '../context/AuthContext';
import { useMeals } from '../context/MealContext';
import {
  FIXED_ACHIEVEMENT_DEFS,
  RANDOM_ACHIEVEMENT_POOL,
  difficultyRank,
  seededShuffle,
} from '../data/achievementsCatalog';

function isValidEmail(value) {
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value || '');
}

function isValidPhone(value) {
  return /^\+?[0-9\-\s()]{7,20}$/.test(value || '');
}

function isStrongPassword(value) {
  return /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[^A-Za-z0-9]).{8,}$/.test(value || '');
}

function resolveAvatarSource(avatar) {
  if (!avatar) return null;
  if (avatar.startsWith('file:') || avatar.startsWith('http')) {
    return { uri: avatar };
  }
  if (avatar.startsWith('/')) {
    return { uri: `${API_BASE_URL}${avatar}` };
  }
  return { uri: avatar };
}

function isLocalImageUri(uri) {
  if (!uri) return false;
  return !(uri.startsWith('http') || uri.startsWith('/'));
}

function mapApiError(error, fallback) {
  const status = Number(error?.status || 0);
  const detail = error?.payload?.detail;
  const detailText = typeof detail === 'string' ? detail : '';

  if (status === 401) return 'Session expired. Please log in again.';
  if (status === 404) {
    return detailText.toLowerCase().includes('user not found')
      ? 'Account not found. Please log in again.'
      : fallback;
  }
  if (status === 405) return 'Unable to update profile right now.';
  if (status === 409) return detailText || 'That username or email is already in use.';
  if (status === 400) return detailText || 'Please review your inputs and try again.';

  if (error?.message && error.message !== 'Not Found') {
    return error.message;
  }
  return fallback;
}

export default function ProfileScreen() {
  const { colors, isDark, setDarkMode } = useTheme();
  const insets = useSafeAreaInsets();
  const navigation = useNavigation();

  const {
    isLoading: authLoading,
    isAuthenticated,
    profile,
    achievements,
    refreshProfile,
    updateProfile,
    uploadAvatar,
    changePassword,
    checkUsernameAvailable,
    logout,
    deleteAccount,
  } = useAuth();
  const { getAchievementMetrics, getXpProgression } = useMeals();

  const [refreshing, setRefreshing] = useState(false);
  const [editOpen, setEditOpen] = useState(false);
  const [saving, setSaving] = useState(false);
  const [formError, setFormError] = useState('');

  const [fullName, setFullName] = useState('');
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [phone, setPhone] = useState('');
  const [country, setCountry] = useState('');
  const [avatarUri, setAvatarUri] = useState('');

  const [currentPassword, setCurrentPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');

  const units = profile?.settings?.units || profile?.unitSystem || 'Metric';
  const notificationsEnabled = profile?.settings?.notifications ?? true;

  useFocusEffect(
    React.useCallback(() => {
      if (authLoading || !isAuthenticated) return;
      refreshProfile().catch(async (error) => {
        if (error?.status === 401) {
          await logout();
        }
      });
    }, [authLoading, isAuthenticated, logout, refreshProfile])
  );

  const stats = useMemo(
    () => [
      { label: 'Day Streak', value: String(profile?.streak || 0), sub: 'Current run', icon: '🔥' },
      { label: 'Total Scans', value: String(profile?.totalScans || 0), sub: 'Meals captured', icon: '📸' },
      { label: 'Member Since', value: profile?.joinedDate || profile?.memberSince || 'Today', sub: 'Joined date', icon: '📅' },
    ],
    [profile]
  );

  const achievementMetrics = useMemo(() => getAchievementMetrics(), [getAchievementMetrics]);
  const xpProgression = useMemo(() => getXpProgression(), [getXpProgression]);

  const achievementStateMap = useMemo(() => {
    const next = new Map();
    (achievements || []).forEach((item) => {
      if (item?.id) next.set(item.id, item);
    });
    return next;
  }, [achievements]);

  const activeRandomDefs = useMemo(() => {
    const today = new Date();
    const dateSeed = Number(`${today.getFullYear()}${today.getMonth() + 1}${today.getDate()}`);
    const userSeed = String(profile?.id || profile?.email || 'guest').split('').reduce((sum, ch) => sum + ch.charCodeAt(0), 0);
    const selected = seededShuffle(RANDOM_ACHIEVEMENT_POOL, dateSeed + userSeed).slice(0, 3);
    return selected.map((item) => ({ ...item, id: `${item.id}_${today.toISOString().slice(0, 10)}`, type: 'random' }));
  }, [profile]);

  const evaluateAchievement = React.useCallback((def) => {
    const rawProgress = Number(achievementMetrics?.[def.metric] || 0);
    const progress = Math.max(0, rawProgress);
    const target = Number(def.target || 1);
    const ratio = target > 0 ? progress / target : 0;
    const persisted = achievementStateMap.get(def.id);
    const completedByProgress = target > 0 ? progress >= target : false;
    const persistedStatus = String(persisted?.status || '').toLowerCase();
    const completedAt = persisted?.completed_at || persisted?.completedAt || (completedByProgress ? new Date().toISOString() : null);

    let status = 'locked';
    if (persistedStatus === 'claimed') status = 'claimed';
    else if (persistedStatus === 'completed' || completedByProgress) status = 'completed';
    else if (progress > 0) status = 'in_progress';

    return {
      ...def,
      progress,
      target,
      ratio,
      status,
      completedAt,
      xpReward: Number(def.xpReward || 0),
      persisted,
    };
  }, [achievementMetrics, achievementStateMap]);

  const fixedAchievements = useMemo(() => FIXED_ACHIEVEMENT_DEFS.map(evaluateAchievement), [evaluateAchievement]);
  const randomAchievements = useMemo(() => activeRandomDefs.map(evaluateAchievement), [activeRandomDefs, evaluateAchievement]);
  const completedAchievements = useMemo(() => [...fixedAchievements, ...randomAchievements].filter((a) => a.status === 'claimed' || a.status === 'completed'), [fixedAchievements, randomAchievements]);
  const featuredAchievements = useMemo(
    () => [...fixedAchievements, ...randomAchievements].sort((a, b) => {
      if (a.status !== b.status) {
        const statusRank = { in_progress: 0, locked: 1, completed: 2, claimed: 3 };
        return (statusRank[a.status] ?? 9) - (statusRank[b.status] ?? 9);
      }
      return difficultyRank(b.difficulty) - difficultyRank(a.difficulty);
    }).slice(0, 4),
    [fixedAchievements, randomAchievements],
  );

  const achievementsXpEarned = useMemo(
    () => [...fixedAchievements, ...randomAchievements]
      .filter((a) => a.status === 'claimed' || a.status === 'completed')
      .reduce((sum, a) => sum + Number(a.xpReward || 0), 0),
    [fixedAchievements, randomAchievements],
  );

  const profileImageSource = resolveAvatarSource(profile?.avatar || avatarUri);

  const openEdit = () => {
    setFormError('');
    setFullName(profile?.name || '');
    setUsername(profile?.username || String(profile?.email || '').split('@')[0] || '');
    setEmail(profile?.email || '');
    setPhone(profile?.phone || '');
    setCountry(profile?.country || '');
    setAvatarUri(profile?.avatar || '');
    setCurrentPassword('');
    setNewPassword('');
    setConfirmPassword('');
    setEditOpen(true);
  };

  const pickAvatar = async () => {
    const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!permission.granted) {
      setFormError('Please allow photo access to update your profile picture.');
      return;
    }

    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      quality: 0.85,
      aspect: [1, 1],
    });

    if (!result.canceled && result.assets?.[0]?.uri) {
      setAvatarUri(result.assets[0].uri);
      setFormError('');
    }
  };

  const onRefresh = async () => {
    try {
      setRefreshing(true);
      await refreshProfile();
    } finally {
      setRefreshing(false);
    }
  };

  const saveEditProfile = async () => {
    try {
      setSaving(true);
      setFormError('');

      const trimmedName = fullName.trim();
      const trimmedUsername = username.trim();
      const trimmedEmail = email.trim();
      const trimmedPhone = phone.trim();

      if (!trimmedName) throw new Error('Full name is required.');
      if (trimmedUsername.length < 3) throw new Error('Username must be at least 3 characters.');
      if (!isValidEmail(trimmedEmail)) throw new Error('Email format is invalid.');
      if (trimmedPhone && !isValidPhone(trimmedPhone)) throw new Error('Phone format is invalid.');

      if (trimmedUsername.toLowerCase() !== String(profile?.username || '').toLowerCase()) {
        const available = await checkUsernameAvailable(trimmedUsername);
        if (!available) throw new Error('Username is already taken.');
      }

      await updateProfile({
        fullName: trimmedName,
        username: trimmedUsername,
        email: trimmedEmail,
        phone: trimmedPhone,
        country: country.trim(),
      });

      if (avatarUri && isLocalImageUri(avatarUri)) {
        await uploadAvatar(avatarUri);
      }

      if (newPassword || confirmPassword || currentPassword) {
        if (!currentPassword) throw new Error('Enter your current password to change it.');
        if (!isStrongPassword(newPassword)) {
          throw new Error('New password must include uppercase, lowercase, number, special character, and be 8+ chars.');
        }
        if (newPassword !== confirmPassword) {
          throw new Error('New password and confirm password do not match.');
        }

        await changePassword({
          currentPassword,
          newPassword,
          confirmPassword,
        });
      }

      // updateProfile already merged and saved state - no refresh needed
      setEditOpen(false);
    } catch (e) {
      console.error('[ProfileScreen Save Error]', {
        message: e?.message,
        status: e?.status,
        payload: e?.payload,
        fullError: e,
      });
      setFormError(mapApiError(e, 'Unable to save profile changes. Check your inputs and try again.'));
    } finally {
      setSaving(false);
    }
  };

  const setNotifications = async (value) => {
    try {
      // updateProfile already handles state merge, no refresh needed
      await updateProfile({ settings: { notifications: value } });
    } catch (e) {
      Alert.alert('Update failed', mapApiError(e, 'Unable to save notification preference right now.'));
    }
  };

  const setUnits = async (nextUnits) => {
    try {
      // updateProfile already handles state merge, no refresh needed
      await updateProfile({ settings: { units: nextUnits } });
    } catch (e) {
      Alert.alert('Update failed', mapApiError(e, 'Unable to save units preference right now.'));
    }
  };

  const handleDarkMode = async (value) => {
    const previousValue = isDark;
    setDarkMode(value);
    try {
      // updateProfile already handles state merge and backend sync
      await updateProfile({ settings: { darkMode: value } });
    } catch (e) {
      setDarkMode(previousValue);
      Alert.alert(
        'Dark Mode Update Failed',
        mapApiError(e, 'Your dark mode change was not saved. Please try again.'),
        [{ text: 'OK', onPress: () => {} }]
      );
    }
  };

  const handleLogout = () => {
    Alert.alert('Log Out', 'Log out from this account?', [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Log Out',
        style: 'destructive',
        onPress: async () => {
          try {
            await logout();
          } catch (e) {
            Alert.alert('Log out failed', mapApiError(e, 'Unable to log out right now.'));
          }
        },
      },
    ]);
  };

  const handleDeleteAccount = () => {
    Alert.alert(
      'Delete Account',
      'This action is permanent. Your account and profile data will be removed.',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Continue',
          style: 'destructive',
          onPress: () => {
            Alert.alert('Confirm Deletion', 'Type-free confirmation: delete your account now?', [
              { text: 'Cancel', style: 'cancel' },
              {
                text: 'Delete Account',
                style: 'destructive',
                onPress: async () => {
                  try {
                    await deleteAccount();
                  } catch (e) {
                    Alert.alert('Delete failed', mapApiError(e, 'Unable to delete account right now.'));
                  }
                },
              },
            ]);
          },
        },
      ]
    );
  };

  return (
    <>
      <ScrollView
        style={[styles.container, { backgroundColor: colors.background }]}
        contentContainerStyle={[styles.content, { paddingTop: insets.top + 8, paddingBottom: insets.bottom + 72 }]}
        showsVerticalScrollIndicator={false}
        refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor={colors.primary} />}
      >
        <Card style={styles.profileCard}>
          <View style={styles.profileTopRow}>
            <View style={styles.profileIdentityRow}>
              <View style={[styles.avatar, { backgroundColor: colors.primary }]}> 
                {profileImageSource ? (
                  <Image source={profileImageSource} style={styles.avatarImage} />
                ) : (
                  <Text style={styles.avatarText}>{profile?.initials || 'U'}</Text>
                )}
              </View>
              <View style={styles.profileIdentityText}>
                <Text style={[styles.profileName, { color: colors.text }]}>{profile?.name || 'User'}</Text>
                <Text numberOfLines={1} style={[styles.profileEmail, { color: colors.textSecondary }]}>{profile?.email || 'No email set'}</Text>
              </View>
            </View>
            <View style={styles.profileEditWrap}>
              <Button title="Edit Profile" size="sm" onPress={openEdit} />
            </View>
          </View>

          <View style={[styles.goalCard, { backgroundColor: colors.primarySoft }]}> 
            <Text style={[styles.goalLabel, { color: colors.primary }]}>GOAL FOCUS</Text>
            <Text style={[styles.goalValue, { color: colors.text }]}>{profile?.goal || 'Not set'}</Text>
          </View>
        </Card>

        <View style={styles.sectionHeader}>
          <Text style={[styles.sectionTitle, { color: colors.text }]}>Your Stats</Text>
          <Text style={[styles.sectionMeta, { color: colors.textSecondary }]}>Live progress</Text>
        </View>

        <View style={styles.statsRow}>
          {stats.map((stat) => (
            <Card key={stat.label} style={styles.statCard}>
              <Text style={styles.statIcon}>{stat.icon}</Text>
              <Text style={[styles.statValue, { color: colors.text }]}>{stat.value}</Text>
              <Text style={[styles.statLabel, { color: colors.textSecondary }]}>{stat.label}</Text>
              <Text style={[styles.statSub, { color: colors.textTertiary }]}>{stat.sub}</Text>
            </Card>
          ))}
        </View>

        <View style={styles.sectionHeader}>
          <Text style={[styles.sectionTitle, { color: colors.text }]}>Achievements</Text>
          <Text style={[styles.sectionMeta, { color: colors.textSecondary }]}>Progress hub</Text>
        </View>

        <TouchableOpacity
          style={styles.achievementsHubTapArea}
          onPress={() => navigation.navigate('Achievements')}
          activeOpacity={0.9}
        >
          <Card style={styles.achievementsSummaryCard}>
            <View style={styles.achievementsHubHeader}>
              <Text style={[styles.achievementsHubTitle, { color: colors.text }]}>Achievements Hub</Text>
            </View>

            <View style={styles.achievementsSummaryRow}>
              <View style={[styles.achievementMiniStat, { backgroundColor: colors.surfaceSecondary }]}>
                <Text style={[styles.achievementMiniLabel, { color: colors.textSecondary }]}>Achievement XP</Text>
                <Text style={[styles.achievementMiniValue, { color: colors.text }]}>{Math.round(achievementsXpEarned)}</Text>
              </View>
              <View style={[styles.achievementMiniStat, { backgroundColor: colors.surfaceSecondary }]}>
                <Text style={[styles.achievementMiniLabel, { color: colors.textSecondary }]}>Completed</Text>
                <Text style={[styles.achievementMiniValue, { color: colors.text }]}>{completedAchievements.length}</Text>
              </View>
              <View style={[styles.achievementMiniStat, { backgroundColor: colors.surfaceSecondary }]}>
                <Text style={[styles.achievementMiniLabel, { color: colors.textSecondary }]}>Active Challenges</Text>
                <Text style={[styles.achievementMiniValue, { color: colors.text }]}>{randomAchievements.length}</Text>
              </View>
            </View>

            <View style={[styles.achievementOpenButton, { backgroundColor: colors.primarySoft }]}>
              <Text style={[styles.achievementOpenButtonText, { color: colors.primary }]}>Open Achievements Hub</Text>
            </View>
          </Card>
        </TouchableOpacity>

        <View style={styles.sectionHeader}>
          <Text style={[styles.sectionTitle, { color: colors.text }]}>Settings</Text>
          <Text style={[styles.sectionMeta, { color: colors.textSecondary }]}>Preferences</Text>
        </View>

        <Card style={styles.settingsCard}>
          <View style={styles.settingRow}>
            <View style={[styles.settingIconCircle, { backgroundColor: colors.surfaceSecondary }]}> 
              <Moon size={20} color={colors.textSecondary} />
            </View>
            <View style={styles.settingInfo}>
              <Text style={[styles.settingLabel, { color: colors.text }]}>Dark Mode</Text>
              <Text style={[styles.settingHint, { color: colors.textSecondary }]}>Theme appearance</Text>
            </View>
            <Switch
              value={isDark}
              onValueChange={handleDarkMode}
              trackColor={{ false: colors.border, true: colors.primaryLight }}
              thumbColor={isDark ? colors.primary : colors.textTertiary}
            />
          </View>

          <View style={[styles.separator, { backgroundColor: colors.borderLight }]} />

          <View style={styles.settingRow}>
            <View style={[styles.settingIconCircle, { backgroundColor: colors.surfaceSecondary }]}> 
              <Bell size={20} color={colors.textSecondary} />
            </View>
            <View style={styles.settingInfo}>
              <Text style={[styles.settingLabel, { color: colors.text }]}>Notifications</Text>
              <Text style={[styles.settingHint, { color: colors.textSecondary }]}>Meal reminders</Text>
            </View>
            <Switch
              value={notificationsEnabled}
              onValueChange={setNotifications}
              trackColor={{ false: colors.border, true: colors.primaryLight }}
              thumbColor={notificationsEnabled ? colors.primary : colors.textTertiary}
            />
          </View>

          <View style={[styles.separator, { backgroundColor: colors.borderLight }]} />

          <View style={styles.settingRow}>
            <View style={[styles.settingIconCircle, { backgroundColor: colors.surfaceSecondary }]}> 
              <Ruler size={20} color={colors.textSecondary} />
            </View>
            <View style={styles.settingInfo}>
              <Text style={[styles.settingLabel, { color: colors.text }]}>Units</Text>
              <Text style={[styles.settingHint, { color: colors.textSecondary }]}>Measurement system</Text>
            </View>
            <View style={styles.unitButtons}>
              {['Metric', 'Imperial'].map((item) => (
                <TouchableOpacity
                  key={item}
                  style={[
                    styles.unitButton,
                    {
                      backgroundColor: units === item ? colors.primarySoft : colors.surfaceSecondary,
                      borderColor: units === item ? colors.primary : colors.border,
                    },
                  ]}
                  onPress={() => setUnits(item)}
                  activeOpacity={0.8}
                >
                  <Text style={[styles.unitButtonText, { color: units === item ? colors.primary : colors.textSecondary }]}>{item}</Text>
                </TouchableOpacity>
              ))}
            </View>
          </View>
        </Card>

        <TouchableOpacity style={[styles.logoutBtn, { borderColor: colors.border }]} onPress={handleLogout} activeOpacity={0.8}>
          <LogOut size={20} color={colors.danger} />
          <Text style={[styles.logoutText, { color: colors.danger }]}>Log Out</Text>
        </TouchableOpacity>

        <TouchableOpacity style={[styles.deleteBtn, { borderColor: colors.danger, backgroundColor: colors.dangerSoft || '#fee2e2' }]} onPress={handleDeleteAccount} activeOpacity={0.8}>
          <Trash2 size={20} color={colors.danger} />
          <Text style={[styles.deleteText, { color: colors.danger }]}>Delete Account</Text>
        </TouchableOpacity>
      </ScrollView>

      <Modal visible={editOpen} transparent animationType="fade" onRequestClose={() => setEditOpen(false)}>
        <View style={[styles.modalOverlay, { paddingTop: insets.top + Spacing.md, paddingBottom: insets.bottom + Spacing.md }]}>
          <View style={[styles.modalCard, { backgroundColor: colors.surface }]}> 
            <ScrollView
              contentContainerStyle={[styles.modalContent, { paddingBottom: insets.bottom + Spacing.lg }]}
              showsVerticalScrollIndicator={false}
              keyboardShouldPersistTaps="handled"
            >
              <View style={styles.modalHeader}>
                <Text style={[styles.modalTitle, { color: colors.text }]}>Edit Profile</Text>
                <TouchableOpacity style={[styles.modalClose, { backgroundColor: colors.surfaceSecondary }]} onPress={() => setEditOpen(false)}>
                  <Text style={[styles.modalCloseText, { color: colors.textSecondary }]}>×</Text>
                </TouchableOpacity>
              </View>
              <Text style={[styles.modalSubtitle, { color: colors.textSecondary }]}>Update your photo and account details.</Text>

              <Text style={[styles.modalSectionLabel, { color: colors.textSecondary }]}>Profile photo</Text>
              <View style={[styles.modalAvatar, { backgroundColor: colors.primarySoft }]}> 
                {avatarUri ? (
                  <Image source={resolveAvatarSource(avatarUri)} style={styles.modalAvatarImage} />
                ) : (
                  <UserRound size={48} color={colors.primary} />
                )}
              </View>
              <Button title="Change Photo" variant="outline" size="sm" onPress={pickAvatar} />

              <Text style={[styles.fieldLabel, { color: colors.textSecondary }]}>Username</Text>
              <TextInput
                style={[styles.input, { borderColor: colors.border, backgroundColor: colors.surfaceSecondary, color: colors.text }]}
                value={username}
                onChangeText={setUsername}
                autoCapitalize="none"
                placeholder="username"
                placeholderTextColor={colors.textTertiary}
              />

              <Text style={[styles.fieldLabel, { color: colors.textSecondary }]}>Full name</Text>
              <TextInput
                style={[styles.input, { borderColor: colors.border, backgroundColor: colors.surfaceSecondary, color: colors.text }]}
                value={fullName}
                onChangeText={setFullName}
                placeholder="Full name"
                placeholderTextColor={colors.textTertiary}
              />

              <Text style={[styles.fieldLabel, { color: colors.textSecondary }]}>Email</Text>
              <TextInput
                style={[styles.input, { borderColor: colors.border, backgroundColor: colors.surfaceSecondary, color: colors.text }]}
                value={email}
                onChangeText={setEmail}
                autoCapitalize="none"
                keyboardType="email-address"
                placeholder="name@email.com"
                placeholderTextColor={colors.textTertiary}
              />

              <Text style={[styles.fieldLabel, { color: colors.textSecondary }]}>Phone</Text>
              <TextInput
                style={[styles.input, { borderColor: colors.border, backgroundColor: colors.surfaceSecondary, color: colors.text }]}
                value={phone}
                onChangeText={setPhone}
                keyboardType="phone-pad"
                placeholder="+1 555 123 4567"
                placeholderTextColor={colors.textTertiary}
              />

              <Text style={[styles.fieldLabel, { color: colors.textSecondary }]}>Country</Text>
              <TextInput
                style={[styles.input, { borderColor: colors.border, backgroundColor: colors.surfaceSecondary, color: colors.text }]}
                value={country}
                onChangeText={setCountry}
                placeholder="Country"
                placeholderTextColor={colors.textTertiary}
              />

              <Text style={[styles.fieldLabel, { color: colors.textSecondary }]}>Goal Focus</Text>
              <TextInput
                style={[
                  styles.input,
                  styles.readonlyInput,
                  { borderColor: colors.border, backgroundColor: colors.surfaceSecondary, color: colors.textSecondary },
                ]}
                value={profile?.goal || ''}
                editable={false}
                selectTextOnFocus={false}
                placeholder="Goal managed from Home"
                placeholderTextColor={colors.textTertiary}
              />
              <Text style={[styles.readonlyHint, { color: colors.textSecondary }]}>Goal is managed from Home -> Your Goal</Text>

              <Text style={[styles.fieldLabel, { color: colors.textSecondary }]}>Current password</Text>
              <TextInput
                style={[styles.input, { borderColor: colors.border, backgroundColor: colors.surfaceSecondary, color: colors.text }]}
                value={currentPassword}
                onChangeText={setCurrentPassword}
                secureTextEntry
                placeholder="Required only to change password"
                placeholderTextColor={colors.textTertiary}
              />

              <Text style={[styles.fieldLabel, { color: colors.textSecondary }]}>New password</Text>
              <TextInput
                style={[styles.input, { borderColor: colors.border, backgroundColor: colors.surfaceSecondary, color: colors.text }]}
                value={newPassword}
                onChangeText={setNewPassword}
                secureTextEntry
                placeholder="At least 8 chars + complexity"
                placeholderTextColor={colors.textTertiary}
              />

              <Text style={[styles.fieldLabel, { color: colors.textSecondary }]}>Confirm new password</Text>
              <TextInput
                style={[styles.input, { borderColor: colors.border, backgroundColor: colors.surfaceSecondary, color: colors.text }]}
                value={confirmPassword}
                onChangeText={setConfirmPassword}
                secureTextEntry
                placeholder="Confirm new password"
                placeholderTextColor={colors.textTertiary}
              />

              {formError ? <Text style={[styles.formError, { color: colors.danger }]}>{formError}</Text> : null}

              <View style={styles.modalActions}>
                <Button title="Cancel" variant="outline" size="sm" onPress={() => setEditOpen(false)} style={styles.modalActionBtn} />
                <Button title={saving ? 'Saving...' : 'Save'} size="sm" onPress={saveEditProfile} disabled={saving} style={styles.modalActionBtn} />
              </View>
            </ScrollView>
          </View>
        </View>
      </Modal>
    </>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  content: { padding: Spacing.lg },
  profileCard: {
    marginBottom: Spacing.lg,
    shadowColor: '#000',
    shadowOpacity: 0.12,
    shadowRadius: 12,
    shadowOffset: { width: 0, height: 6 },
    elevation: 5,
  },
  profileTopRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: Spacing.sm,
  },
  profileIdentityRow: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
    gap: Spacing.md,
  },
  profileIdentityText: {
    flex: 1,
    paddingRight: Spacing.xs,
  },
  profileEditWrap: {
    paddingTop: 6,
  },
  avatar: {
    width: 108,
    height: 108,
    borderRadius: 54,
    alignItems: 'center',
    justifyContent: 'center',
    overflow: 'hidden',
  },
  avatarImage: {
    width: '100%',
    height: '100%',
    resizeMode: 'cover',
  },
  avatarText: {
    ...Typography.h1,
    color: '#fff',
  },
  profileName: {
    ...Typography.h1,
    marginBottom: 1,
  },
  profileEmail: {
    ...Typography.body,
    marginTop: 1,
    maxWidth: '100%',
  },
  goalCard: {
    borderRadius: BorderRadius.lg,
    padding: Spacing.md,
  },
  goalLabel: {
    ...Typography.captionMedium,
    marginBottom: 2,
  },
  goalValue: {
    ...Typography.h2,
    fontSize: 24,
  },
  sectionHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: Spacing.sm,
    marginTop: Spacing.xs,
  },
  sectionTitle: {
    ...Typography.h1,
  },
  sectionMeta: {
    ...Typography.body,
  },
  statsRow: {
    flexDirection: 'row',
    gap: Spacing.sm,
    marginBottom: Spacing.lg,
  },
  statCard: {
    flex: 1,
    minHeight: 154,
    alignItems: 'center',
    justifyContent: 'center',
    gap: 2,
    paddingVertical: Spacing.md,
    paddingHorizontal: Spacing.xs,
  },
  statIcon: {
    fontSize: 18,
    marginBottom: 3,
  },
  statValue: {
    ...Typography.h2,
    marginBottom: 2,
  },
  statLabel: {
    ...Typography.bodyMedium,
    textAlign: 'center',
    lineHeight: 26,
  },
  statSub: {
    ...Typography.caption,
    textAlign: 'center',
    lineHeight: 23,
  },
  achievementList: {
    gap: Spacing.md,
    marginBottom: Spacing.xl,
  },
  achievementsSummaryCard: {
    marginBottom: Spacing.md,
    padding: Spacing.md,
  },
  achievementsSummaryRow: {
    flexDirection: 'row',
    gap: 8,
    marginBottom: Spacing.md,
  },
  achievementMiniStat: {
    flex: 1,
    borderRadius: BorderRadius.md,
    paddingVertical: Spacing.sm,
    paddingHorizontal: 6,
    alignItems: 'center',
  },
  achievementMiniLabel: {
    ...Typography.caption,
    marginBottom: 4,
    textAlign: 'center',
    lineHeight: 20,
  },
  achievementMiniValue: {
    ...Typography.h2,
    fontSize: 17,
  },
  achievementHintText: {
    ...Typography.body,
  },
  achievementsHubTapArea: {
    marginBottom: Spacing.xl,
  },
  achievementsHubHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'flex-start',
    marginBottom: Spacing.md,
  },
  achievementsHubTitle: {
    ...Typography.h2,
  },
  achievementsHubCta: {
    ...Typography.bodyMedium,
    fontWeight: '700',
  },
  achievementOpenButton: {
    marginTop: Spacing.xs,
    borderRadius: BorderRadius.md,
    minHeight: 44,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: Spacing.md,
  },
  achievementOpenButtonText: {
    ...Typography.bodyMedium,
    fontWeight: '700',
  },
  achievementSectionBlock: {
    marginBottom: Spacing.lg,
  },
  achievementSectionTitle: {
    ...Typography.h2,
    marginBottom: Spacing.sm,
  },
  achievementCard: {
    gap: Spacing.xs,
  },
  achievementRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: Spacing.xs,
  },
  achievementIconCircle: {
    width: 44,
    height: 44,
    borderRadius: 22,
    alignItems: 'center',
    justifyContent: 'center',
  },
  achievementEmoji: {
    fontSize: 22,
  },
  lockBadge: {
    borderRadius: BorderRadius.full,
    paddingHorizontal: Spacing.md,
    paddingVertical: 6,
  },
  lockText: {
    ...Typography.captionMedium,
  },
  achievementTitle: {
    ...Typography.h3,
    marginBottom: 2,
  },
  achievementDescription: {
    ...Typography.body,
  },
  categoryTag: {
    alignSelf: 'flex-start',
    marginTop: Spacing.xs,
    borderRadius: BorderRadius.full,
    paddingHorizontal: Spacing.md,
    paddingVertical: 6,
  },
  achievementMetaRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 6,
    marginTop: Spacing.xs,
  },
  categoryText: {
    ...Typography.caption,
  },
  achievementProgressText: {
    ...Typography.caption,
    marginTop: Spacing.xs,
  },
  achievementProgressTrack: {
    height: 6,
    borderRadius: BorderRadius.full,
    marginTop: 4,
    overflow: 'hidden',
  },
  achievementProgressFill: {
    height: '100%',
    borderRadius: BorderRadius.full,
  },
  claimButton: {
    marginTop: Spacing.sm,
    minHeight: 38,
    borderRadius: BorderRadius.md,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: Spacing.md,
  },
  claimButtonText: {
    ...Typography.bodyMedium,
    color: '#ffffff',
    fontWeight: '700',
  },
  achievementCardCompact: {
    paddingVertical: Spacing.sm,
  },
  achievementCompactRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.sm,
  },
  achievementCompactEmoji: {
    fontSize: 20,
  },
  achievementCompactInfo: {
    flex: 1,
  },
  achievementCompactTitle: {
    ...Typography.bodyMedium,
    fontWeight: '700',
  },
  achievementCompactMeta: {
    ...Typography.caption,
  },
  achievementCompactStatus: {
    ...Typography.captionMedium,
  },
  claimButtonMini: {
    minHeight: 30,
    borderRadius: BorderRadius.full,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: Spacing.md,
  },
  claimButtonMiniText: {
    ...Typography.captionMedium,
    color: '#ffffff',
  },
  settingsCard: {
    marginBottom: Spacing.lg,
    padding: 0,
  },
  settingRow: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: Spacing.lg,
    gap: Spacing.md,
  },
  settingIconCircle: {
    width: 42,
    height: 42,
    borderRadius: 21,
    alignItems: 'center',
    justifyContent: 'center',
  },
  settingInfo: {
    flex: 1,
  },
  settingLabel: {
    ...Typography.bodyMedium,
    fontWeight: '700',
  },
  settingHint: {
    ...Typography.body,
  },
  separator: {
    height: 1,
    marginHorizontal: Spacing.lg,
  },
  unitButtons: {
    flexDirection: 'row',
    gap: 6,
  },
  unitButton: {
    borderWidth: 1,
    borderRadius: BorderRadius.full,
    paddingHorizontal: Spacing.sm,
    paddingVertical: 6,
  },
  unitButtonText: {
    ...Typography.caption,
    fontWeight: '700',
  },
  logoutBtn: {
    borderWidth: 1,
    borderRadius: BorderRadius.lg,
    paddingVertical: Spacing.lg,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: Spacing.sm,
    marginBottom: Spacing.md,
  },
  logoutText: {
    ...Typography.h3,
  },
  deleteBtn: {
    borderWidth: 1,
    borderRadius: BorderRadius.lg,
    paddingVertical: Spacing.lg,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: Spacing.sm,
  },
  deleteText: {
    ...Typography.h3,
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.38)',
    justifyContent: 'center',
    padding: Spacing.lg,
  },
  modalCard: {
    maxHeight: '90%',
    borderRadius: BorderRadius.xl,
    overflow: 'hidden',
  },
  modalContent: {
    padding: Spacing.lg,
    gap: Spacing.sm,
  },
  modalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  modalTitle: {
    ...Typography.h2,
  },
  modalClose: {
    width: 36,
    height: 36,
    borderRadius: 18,
    alignItems: 'center',
    justifyContent: 'center',
  },
  modalCloseText: {
    fontSize: 26,
    lineHeight: 26,
  },
  modalSubtitle: {
    ...Typography.body,
    marginBottom: Spacing.xs,
  },
  modalSectionLabel: {
    ...Typography.captionMedium,
    textAlign: 'center',
  },
  modalAvatar: {
    width: 140,
    height: 140,
    borderRadius: 70,
    alignSelf: 'center',
    alignItems: 'center',
    justifyContent: 'center',
    overflow: 'hidden',
    marginBottom: Spacing.xs,
  },
  modalAvatarImage: {
    width: '100%',
    height: '100%',
    resizeMode: 'cover',
  },
  fieldLabel: {
    ...Typography.captionMedium,
    marginTop: 2,
  },
  input: {
    borderWidth: 1,
    borderRadius: BorderRadius.md,
    paddingHorizontal: Spacing.md,
    paddingVertical: Spacing.md,
    ...Typography.body,
  },
  readonlyInput: {
    opacity: 0.72,
  },
  readonlyHint: {
    ...Typography.caption,
    marginTop: 4,
  },
  formError: {
    ...Typography.caption,
    marginTop: Spacing.xs,
  },
  modalActions: {
    flexDirection: 'row',
    gap: Spacing.sm,
    marginTop: Spacing.md,
  },
  modalActionBtn: {
    flex: 1,
  },
});
