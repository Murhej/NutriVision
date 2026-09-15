import React, { useState, useMemo } from 'react';
import { View, Text, StyleSheet, ScrollView, TouchableOpacity } from 'react-native';
import { useTheme } from '../theme/ThemeContext';
import { Typography, Spacing, BorderRadius, Shadow } from '../theme';
import { Card } from '../components';
import { ActivityIndicator } from 'react-native';
import { useFocusEffect } from '@react-navigation/native';
import { apiClient } from '../api/client';

export default function LeaderboardScreen() {
  const { colors } = useTheme();
  const [period, setPeriod] = useState('weekly');
  const [LEADERBOARD_USERS, setLeaderboardUsers] = useState([]);
  const [loading, setLoading] = useState(true);

  useFocusEffect(
    React.useCallback(() => {
      const fetchLeaderboard = async () => {
        try {
          const res = await apiClient.get('/api/mobile/leaderboard');
          setLeaderboardUsers(res.users);
        } catch (error) {
          console.error('Error fetching leaderboard:', error);
        } finally {
          setLoading(false);
        }
      };
      fetchLeaderboard();
    }, [])
  );

  // Sort users by score descending
  const sorted = useMemo(() => {
    const users = [...LEADERBOARD_USERS].sort((a, b) => b.score - a.score);
    return users;
  }, [LEADERBOARD_USERS]);

  if (loading) {
    return (
      <View style={[styles.container, { backgroundColor: colors.background, justifyContent: 'center', alignItems: 'center' }]}>
        <ActivityIndicator size="large" color={colors.primary} />
      </View>
    );
  }

  const podium = sorted.slice(0, 3);
  const rest = sorted.slice(3);

  const podiumColors = ['#f59e0b', '#94a3b8', '#cd7f32']; // gold, silver, bronze
  const podiumLabels = ['🥇', '🥈', '🥉'];

  return (
    <ScrollView
      style={[styles.container, { backgroundColor: colors.background }]}
      contentContainerStyle={styles.content}
      showsVerticalScrollIndicator={false}
    >
      {/* Period Toggle */}
      <View style={[styles.toggle, { backgroundColor: colors.surfaceSecondary }]}>
        {['weekly', 'monthly'].map((p) => (
          <TouchableOpacity
            key={p}
            style={[
              styles.toggleBtn,
              period === p && { backgroundColor: colors.surface },
              period === p && Shadow.sm,
              period === p && { shadowColor: colors.shadowColor, shadowOpacity: colors.shadowOpacity },
            ]}
            onPress={() => setPeriod(p)}
            activeOpacity={0.7}
          >
            <Text style={[
              styles.toggleText,
              { color: period === p ? colors.text : colors.textSecondary },
            ]}>
              {p === 'weekly' ? 'This Week' : 'This Month'}
            </Text>
          </TouchableOpacity>
        ))}
      </View>

      {/* Podium */}
      <View style={styles.podiumRow}>
        {[1, 0, 2].map((podiumIndex) => {
          const user = podium[podiumIndex];
          if (!user) return null;
          const isFirst = podiumIndex === 0;
          return (
            <View key={user.id} style={[styles.podiumItem, isFirst && styles.podiumFirst]}>
              <Text style={styles.podiumMedal}>{podiumLabels[podiumIndex]}</Text>
              <View
                style={[
                  styles.podiumAvatar,
                  {
                    backgroundColor: podiumColors[podiumIndex],
                    width: isFirst ? 64 : 52,
                    height: isFirst ? 64 : 52,
                    borderRadius: isFirst ? 32 : 26,
                  },
                ]}
              >
                <Text style={[styles.podiumInitials, isFirst && { fontSize: 22 }]}>
                  {user.initials}
                </Text>
              </View>
              <Text style={[styles.podiumName, { color: colors.text }]} numberOfLines={1}>
                {user.name.split(' ')[0]}
              </Text>
              <Text style={[styles.podiumScore, { color: colors.textSecondary }]}>
                {user.score.toLocaleString()} pts
              </Text>
            </View>
          );
        })}
      </View>

      {/* Rankings List */}
      <Card style={styles.rankingsCard}>
        {rest.map((user, index) => {
          const rank = index + 4;
          return (
            <View key={user.id}>
              <View
                style={[
                  styles.rankRow,
                  user.isCurrentUser && { backgroundColor: colors.primarySoft },
                  user.isCurrentUser && { marginHorizontal: -Spacing.lg, paddingHorizontal: Spacing.lg },
                ]}
              >
                <Text style={[styles.rankNum, { color: colors.textTertiary }]}>{rank}</Text>
                <View
                  style={[
                    styles.rankAvatar,
                    {
                      backgroundColor: user.isCurrentUser ? colors.primary : colors.surfaceSecondary,
                    },
                  ]}
                >
                  <Text
                    style={[
                      styles.rankInitials,
                      { color: user.isCurrentUser ? '#fff' : colors.textSecondary },
                    ]}
                  >
                    {user.initials}
                  </Text>
                </View>
                <View style={styles.rankInfo}>
                  <Text style={[styles.rankName, { color: colors.text }]}>
                    {user.name}
                    {user.isCurrentUser ? ' (You)' : ''}
                  </Text>
                  <Text style={[styles.rankStreak, { color: colors.textSecondary }]}>
                    🔥 {user.streak} day streak
                  </Text>
                </View>
                <Text style={[styles.rankScore, { color: colors.text }]}>
                  {user.score.toLocaleString()}
                </Text>
              </View>
              {index < rest.length - 1 && (
                <View style={[styles.separator, { backgroundColor: colors.borderLight }]} />
              )}
            </View>
          );
        })}
      </Card>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  content: { padding: Spacing.lg, paddingBottom: Spacing.xxxl * 2 },
  toggle: {
    flexDirection: 'row',
    borderRadius: BorderRadius.lg,
    padding: Spacing.xs,
    marginBottom: Spacing.xxl,
  },
  toggleBtn: {
    flex: 1,
    paddingVertical: Spacing.sm,
    borderRadius: BorderRadius.md,
    alignItems: 'center',
  },
  toggleText: {
    ...Typography.captionMedium,
  },
  podiumRow: {
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'flex-end',
    marginBottom: Spacing.xxl,
    gap: Spacing.md,
  },
  podiumItem: {
    alignItems: 'center',
    gap: Spacing.xs,
    flex: 1,
  },
  podiumFirst: {
    marginBottom: Spacing.lg,
  },
  podiumMedal: {
    fontSize: 24,
  },
  podiumAvatar: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  podiumInitials: {
    fontSize: 18,
    fontWeight: '700',
    color: '#fff',
  },
  podiumName: {
    ...Typography.captionMedium,
    textAlign: 'center',
  },
  podiumScore: {
    ...Typography.caption,
  },
  rankingsCard: {
    padding: 0,
    paddingVertical: Spacing.sm,
  },
  rankRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: Spacing.md,
    paddingHorizontal: Spacing.lg,
    gap: Spacing.md,
  },
  rankNum: {
    ...Typography.bodyMedium,
    width: 24,
    textAlign: 'center',
  },
  rankAvatar: {
    width: 40,
    height: 40,
    borderRadius: 20,
    alignItems: 'center',
    justifyContent: 'center',
  },
  rankInitials: {
    ...Typography.captionMedium,
    fontWeight: '700',
  },
  rankInfo: {
    flex: 1,
    gap: 2,
  },
  rankName: {
    ...Typography.bodyMedium,
  },
  rankStreak: {
    ...Typography.caption,
  },
  rankScore: {
    ...Typography.bodyMedium,
    fontWeight: '700',
  },
  separator: {
    height: 1,
    marginHorizontal: Spacing.lg,
  },
});
