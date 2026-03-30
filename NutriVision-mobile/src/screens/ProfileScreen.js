import React from 'react';
import { View, Text, StyleSheet, ScrollView, TouchableOpacity, Switch } from 'react-native';
import { useTheme } from '../theme/ThemeContext';
import { Typography, Spacing, BorderRadius, Shadow } from '../theme';
import { Card, ProgressBar } from '../components';
import { USER_PROFILE, ACHIEVEMENTS } from '../data/mockData';
import { ChevronRight, Moon, Bell, Ruler, LogOut } from 'lucide-react-native';

export default function ProfileScreen() {
  const { colors, isDark, toggleTheme } = useTheme();

  const stats = [
    { label: 'Streak', value: `${USER_PROFILE.streak}`, emoji: '🔥' },
    { label: 'Scans', value: `${USER_PROFILE.totalScans}`, emoji: '📸' },
    { label: 'Member Since', value: USER_PROFILE.memberSince, emoji: '📅' },
  ];

  const settingsItems = [
    { icon: <Moon size={20} color={colors.textSecondary} />, label: 'Dark Mode', toggle: true, value: isDark, onToggle: toggleTheme },
    { icon: <Bell size={20} color={colors.textSecondary} />, label: 'Notifications', toggle: true, value: true, onToggle: () => {} },
    { icon: <Ruler size={20} color={colors.textSecondary} />, label: 'Units', detail: 'Metric', toggle: false },
  ];

  return (
    <ScrollView
      style={[styles.container, { backgroundColor: colors.background }]}
      contentContainerStyle={styles.content}
      showsVerticalScrollIndicator={false}
    >
      {/* Profile Header */}
      <View style={styles.profileHeader}>
        <View style={[styles.avatar, { backgroundColor: colors.primary }]}>
          <Text style={styles.avatarText}>{USER_PROFILE.initials}</Text>
        </View>
        <Text style={[styles.profileName, { color: colors.text }]}>{USER_PROFILE.name}</Text>
        <Text style={[styles.profileGoal, { color: colors.textSecondary }]}>
          Goal: {USER_PROFILE.goal}
        </Text>
        <TouchableOpacity
          style={[styles.editBtn, { borderColor: colors.border }]}
          activeOpacity={0.7}
        >
          <Text style={[styles.editBtnText, { color: colors.primary }]}>Edit Profile</Text>
        </TouchableOpacity>
      </View>

      {/* Stats Row */}
      <View style={styles.statsRow}>
        {stats.map((stat, index) => (
          <Card key={index} style={styles.statCard}>
            <Text style={styles.statEmoji}>{stat.emoji}</Text>
            <Text style={[styles.statValue, { color: colors.text }]}>{stat.value}</Text>
            <Text style={[styles.statLabel, { color: colors.textSecondary }]}>{stat.label}</Text>
          </Card>
        ))}
      </View>

      {/* Achievements */}
      <Text style={[styles.sectionTitle, { color: colors.text }]}>Achievements</Text>
      <View style={styles.achievementsGrid}>
        {ACHIEVEMENTS.map((achievement) => (
          <Card
            key={achievement.id}
            style={[
              styles.achievementCard,
              !achievement.unlocked && { opacity: 0.45 },
            ]}
          >
            <Text style={styles.achievementEmoji}>{achievement.emoji}</Text>
            <Text style={[styles.achievementTitle, { color: colors.text }]} numberOfLines={1}>
              {achievement.title}
            </Text>
            <Text style={[styles.achievementDesc, { color: colors.textSecondary }]} numberOfLines={2}>
              {achievement.description}
            </Text>
          </Card>
        ))}
      </View>

      {/* Settings */}
      <Text style={[styles.sectionTitle, { color: colors.text }]}>Settings</Text>
      <Card style={styles.settingsCard}>
        {settingsItems.map((item, index) => (
          <View key={index}>
            <View style={styles.settingsRow}>
              {item.icon}
              <Text style={[styles.settingsLabel, { color: colors.text }]}>{item.label}</Text>
              {item.toggle ? (
                <Switch
                  value={item.value}
                  onValueChange={item.onToggle}
                  trackColor={{ false: colors.border, true: colors.primaryLight }}
                  thumbColor={item.value ? colors.primary : colors.textTertiary}
                />
              ) : (
                <View style={styles.settingsDetail}>
                  <Text style={[styles.settingsDetailText, { color: colors.textSecondary }]}>
                    {item.detail}
                  </Text>
                  <ChevronRight size={16} color={colors.textTertiary} />
                </View>
              )}
            </View>
            {index < settingsItems.length - 1 && (
              <View style={[styles.separator, { backgroundColor: colors.borderLight }]} />
            )}
          </View>
        ))}
      </Card>

      {/* Log Out */}
      <TouchableOpacity
        style={[styles.logoutBtn, { borderColor: colors.danger }]}
        activeOpacity={0.7}
      >
        <LogOut size={18} color={colors.danger} />
        <Text style={[styles.logoutText, { color: colors.danger }]}>Log Out</Text>
      </TouchableOpacity>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  content: { padding: Spacing.lg, paddingBottom: Spacing.xxxl * 2 },
  profileHeader: {
    alignItems: 'center',
    marginBottom: Spacing.xxl,
    gap: Spacing.sm,
  },
  avatar: {
    width: 80,
    height: 80,
    borderRadius: 40,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: Spacing.sm,
  },
  avatarText: {
    fontSize: 28,
    fontWeight: '700',
    color: '#ffffff',
  },
  profileName: {
    ...Typography.h1,
  },
  profileGoal: {
    ...Typography.body,
  },
  editBtn: {
    borderWidth: 1,
    borderRadius: BorderRadius.full,
    paddingHorizontal: Spacing.xl,
    paddingVertical: Spacing.sm,
    marginTop: Spacing.sm,
  },
  editBtnText: {
    ...Typography.captionMedium,
  },
  statsRow: {
    flexDirection: 'row',
    gap: Spacing.md,
    marginBottom: Spacing.xxl,
  },
  statCard: {
    flex: 1,
    alignItems: 'center',
    padding: Spacing.md,
    gap: Spacing.xs,
  },
  statEmoji: { fontSize: 22 },
  statValue: { ...Typography.h3 },
  statLabel: { ...Typography.small, textTransform: 'none' },
  sectionTitle: {
    ...Typography.h3,
    marginBottom: Spacing.md,
  },
  achievementsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: Spacing.md,
    marginBottom: Spacing.xxl,
  },
  achievementCard: {
    width: '30%',
    alignItems: 'center',
    padding: Spacing.md,
    gap: Spacing.xs,
  },
  achievementEmoji: { fontSize: 28 },
  achievementTitle: { ...Typography.captionMedium, textAlign: 'center' },
  achievementDesc: { ...Typography.caption, textAlign: 'center', fontSize: 10, lineHeight: 14 },
  settingsCard: {
    padding: 0,
    marginBottom: Spacing.xl,
  },
  settingsRow: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: Spacing.lg,
    gap: Spacing.md,
  },
  settingsLabel: {
    ...Typography.bodyMedium,
    flex: 1,
  },
  settingsDetail: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.xs,
  },
  settingsDetailText: {
    ...Typography.caption,
  },
  separator: {
    height: 1,
    marginHorizontal: Spacing.lg,
  },
  logoutBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: Spacing.sm,
    padding: Spacing.lg,
    borderRadius: BorderRadius.lg,
    borderWidth: 1,
  },
  logoutText: {
    ...Typography.bodyMedium,
    fontWeight: '700',
  },
});
