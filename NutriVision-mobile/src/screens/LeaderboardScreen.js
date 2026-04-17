import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { ActivityIndicator, Image, RefreshControl, ScrollView, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useTheme } from '../theme/ThemeContext';
import { Typography, Spacing, BorderRadius, Shadow } from '../theme';
import { Card } from '../components';
import { useAuth } from '../context/AuthContext';
import { useMeals } from '../context/MealContext';
import { API_BASE_URL } from '../api/client';

const SCOPES = [
  { id: 'worldwide', label: 'Worldwide' },
  { id: 'canada', label: 'Canada' },
];

function toAvatarSource(avatar) {
  if (!avatar) return null;
  if (avatar.startsWith('http') || avatar.startsWith('file:')) return { uri: avatar };
  if (avatar.startsWith('/')) return { uri: `${API_BASE_URL}${avatar}` };
  return { uri: avatar };
}

function normalizeEntry(entry) {
  const streak = Number.isFinite(Number(entry?.streak)) ? Number(entry.streak) : 0;
  const scans = Number.isFinite(Number(entry?.totalScans)) ? Number(entry.totalScans) : 0;
  const points = Number.isFinite(Number(entry?.points)) ? Number(entry.points) : 0;
  return {
    ...entry,
    streak,
    totalScans: scans,
    points,
    rank: Number(entry?.rank || 0),
    name: entry?.name || 'User',
    initials: entry?.initials || 'U',
  };
}

function PodiumCard({ entry, colors, place }) {
  if (!entry) return null;
  const avatarSource = toAvatarSource(entry.avatar);
  const isFirst = place === 1;
  return (
    <Card
      style={[
        styles.podiumCard,
        isFirst && styles.podiumFirst,
        isFirst && { borderWidth: 1.5, borderColor: colors.primaryLight, backgroundColor: colors.primarySoft },
      ]}
    >
      <Text style={[styles.placeLabel, { color: isFirst ? colors.primary : colors.textSecondary }]}>#{place}</Text>
      <View style={[styles.podiumAvatar, { backgroundColor: isFirst ? colors.primarySoft : colors.surfaceSecondary }]}>
        {avatarSource ? (
          <Image source={avatarSource} style={styles.podiumAvatarImage} />
        ) : (
          <Text style={[styles.podiumInitials, { color: isFirst ? colors.primary : colors.text }]}>{entry.initials}</Text>
        )}
      </View>
      <Text style={[styles.podiumName, { color: colors.text }]} numberOfLines={1}>{entry.name}</Text>
      <Text style={[styles.podiumPoints, { color: colors.textSecondary }]}>{entry.points} pts</Text>
      <Text style={[styles.podiumMeta, { color: colors.textSecondary }]}>{entry.streak} day streak</Text>
    </Card>
  );
}

export default function LeaderboardScreen() {
  const { colors } = useTheme();
  const insets = useSafeAreaInsets();
  const { fetchLeaderboard, profile, achievements } = useAuth();
  const { getXpProgression } = useMeals();

  const [scope, setScope] = useState('worldwide');
  const [users, setUsers] = useState([]);
  const [currentUser, setCurrentUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState('');

  const loadLeaderboard = useCallback(async (selectedScope = scope, isRefresh = false) => {
    try {
      if (isRefresh) {
        setRefreshing(true);
      } else {
        setLoading(true);
      }
      setError('');
      const data = await fetchLeaderboard(selectedScope);
      const normalized = (Array.isArray(data?.users) ? data.users : []).map(normalizeEntry);
      setUsers(normalized);
      setCurrentUser(data?.currentUser ? normalizeEntry(data.currentUser) : null);
    } catch (e) {
      if (e?.status === 404) {
        setUsers([]);
        setCurrentUser(null);
        setError('');
      } else {
        setError(e?.message || 'Unable to load leaderboard.');
      }
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, [fetchLeaderboard, scope]);

  useEffect(() => {
    loadLeaderboard('worldwide');
  }, [loadLeaderboard]);

  const handleScopeChange = async (nextScope) => {
    setScope(nextScope);
    await loadLeaderboard(nextScope);
  };

  const xpProgression = useMemo(() => getXpProgression(), [getXpProgression]);
  const localXp = useMemo(() => Math.round(Number(xpProgression?.totalXp || 0)), [xpProgression]);
  const localStreak = useMemo(() => Math.round(Number(xpProgression?.streak || 0)), [xpProgression]);

  const displayedUsers = useMemo(() => {
    const currentUserId = profile?.id || profile?.user_id || profile?.userId || null;
    const merged = (users || []).map((entry) => {
      const isSelf = Boolean(entry?.isCurrentUser) || (currentUserId && String(entry?.id) === String(currentUserId));
      if (!isSelf) return { ...entry };

      return {
        ...entry,
        points: Math.max(Number(entry?.points || 0), localXp),
        streak: Math.max(Number(entry?.streak || 0), localStreak),
      };
    });

    const sorted = merged.sort((a, b) => {
      if (Number(b.points || 0) !== Number(a.points || 0)) return Number(b.points || 0) - Number(a.points || 0);
      if (Number(b.streak || 0) !== Number(a.streak || 0)) return Number(b.streak || 0) - Number(a.streak || 0);
      return Number(b.totalScans || 0) - Number(a.totalScans || 0);
    });

    return sorted.map((entry, index) => ({ ...entry, rank: index + 1 }));
  }, [localStreak, localXp, profile, users]);

  const { topThree, remainingUsers } = useMemo(() => ({
    topThree: displayedUsers.slice(0, 3),
    remainingUsers: displayedUsers.slice(3),
  }), [displayedUsers]);

  const syncedCurrentUser = useMemo(() => {
    const currentUserId = profile?.id || profile?.user_id || profile?.userId || null;
    const fromList = displayedUsers.find(
      (entry) => Boolean(entry?.isCurrentUser) || (currentUserId && String(entry?.id) === String(currentUserId)),
    );
    if (fromList) return fromList;

    if (!currentUser) return null;
    return {
      ...currentUser,
      points: Math.max(Number(currentUser.points || 0), localXp),
      streak: Math.max(Number(currentUser.streak || 0), localStreak),
    };
  }, [currentUser, displayedUsers, localStreak, localXp, profile]);

  useEffect(() => {
    const refreshTimer = setTimeout(() => {
      loadLeaderboard(scope, true);
    }, 250);

    return () => clearTimeout(refreshTimer);
  }, [achievements, localStreak, localXp, loadLeaderboard, scope]);

  return (
    <ScrollView
      style={[styles.container, { backgroundColor: colors.background }]}
      contentContainerStyle={[styles.content, { paddingTop: insets.top + 8, paddingBottom: insets.bottom + 72 }]}
      showsVerticalScrollIndicator={false}
      refreshControl={<RefreshControl refreshing={refreshing} onRefresh={() => loadLeaderboard(scope, true)} tintColor={colors.primary} />}
    >
      <Card style={styles.scopeCard}>
        <View style={styles.scopeHeader}>
          <View>
            <Text style={[styles.scopeTag, { color: colors.primary }]}>LEADERBOARD</Text>
            <Text style={[styles.scopeTitle, { color: colors.text }]}>Leaderboard Scope</Text>
          </View>
          <View style={[styles.scopeBadge, { backgroundColor: colors.primarySoft }]}> 
            <Text style={[styles.scopeBadgeText, { color: colors.primary }]}>{scope === 'worldwide' ? 'Worldwide' : 'Canada'}</Text>
          </View>
        </View>

        <View style={[styles.scopeToggle, { backgroundColor: colors.surfaceSecondary }]}> 
          {SCOPES.map((item) => (
            <TouchableOpacity
              key={item.id}
              style={[
                styles.scopeButton,
                {
                  backgroundColor: scope === item.id ? colors.surface : 'transparent',
                  borderColor: scope === item.id ? colors.primaryLight : 'transparent',
                },
                scope === item.id && Shadow.sm,
                scope === item.id && { shadowColor: colors.shadowColor, shadowOpacity: colors.shadowOpacity },
              ]}
              onPress={() => handleScopeChange(item.id)}
              activeOpacity={0.75}
            >
              <Text style={[styles.scopeButtonText, { color: scope === item.id ? colors.text : colors.textSecondary }]}>{item.label}</Text>
            </TouchableOpacity>
          ))}
        </View>
      </Card>

      {loading ? (
        <View style={styles.centerBlock}>
          <ActivityIndicator size="large" color={colors.primary} />
        </View>
      ) : error ? (
        <Card style={styles.stateCard}>
          <Text style={[styles.stateTitle, { color: colors.text }]}>Unable to load leaderboard</Text>
          <Text style={[styles.stateSubtitle, { color: colors.textSecondary }]}>{error}</Text>
        </Card>
      ) : users.length === 0 ? (
        <Card style={styles.stateCard}>
          <Text style={[styles.stateTitle, { color: colors.text }]}>No leaderboard data yet</Text>
          <Text style={[styles.stateSubtitle, { color: colors.textSecondary }]}>Rankings appear after real activity data is added.</Text>
        </Card>
      ) : (
        <>
          <View style={styles.podiumLayout}>
            <View style={styles.podiumSide}><PodiumCard entry={topThree[1]} colors={colors} place={2} /></View>
            <View style={styles.podiumCenter}><PodiumCard entry={topThree[0]} colors={colors} place={1} /></View>
            <View style={styles.podiumSide}><PodiumCard entry={topThree[2]} colors={colors} place={3} /></View>
          </View>

          {remainingUsers.length > 0 && (
            <Card style={styles.listCard}>
              {remainingUsers.map((entry, index) => {
                const avatarSource = toAvatarSource(entry.avatar);
                return (
                  <View key={entry.id}>
                    <View style={styles.row}>
                      <Text style={[styles.rank, { color: colors.textTertiary }]}>{entry.rank}</Text>
                      <View style={[styles.avatar, { backgroundColor: colors.surfaceSecondary }]}> 
                        {avatarSource ? (
                          <Image source={avatarSource} style={styles.avatarImage} />
                        ) : (
                          <Text style={[styles.initials, { color: colors.text }]}>{entry.initials}</Text>
                        )}
                      </View>
                      <View style={styles.userInfo}>
                        <Text style={[styles.userName, { color: colors.text }]} numberOfLines={1}>{entry.name}</Text>
                        <Text style={[styles.userMeta, { color: colors.textSecondary }]}>
                          {entry.streak} day streak · {entry.totalScans} scans
                        </Text>
                      </View>
                      <Text style={[styles.points, { color: colors.text }]}>{entry.points}</Text>
                    </View>
                    {index < remainingUsers.length - 1 && <View style={[styles.separator, { backgroundColor: colors.borderLight }]} />}
                  </View>
                );
              })}
            </Card>
          )}
        </>
      )}

      <Card style={[styles.currentUserCard, { borderColor: colors.primaryLight, backgroundColor: colors.primarySoft }]}> 
        <Text style={[styles.currentUserTag, { color: colors.textSecondary }]}>YOUR RANK</Text>
        <View style={styles.currentUserRow}>
          <View style={[styles.rankPill, { backgroundColor: colors.primarySoft }]}> 
            <Text style={[styles.rankPillText, { color: colors.primary }]}>{syncedCurrentUser?.rank ? `#${syncedCurrentUser.rank}` : '--'}</Text>
          </View>
          <View style={[styles.currentUserAvatar, { backgroundColor: colors.primary }]}> 
            {toAvatarSource(syncedCurrentUser?.avatar) ? (
              <Image source={toAvatarSource(syncedCurrentUser?.avatar)} style={styles.currentUserAvatarImage} />
            ) : (
              <Text style={styles.currentUserInitials}>{syncedCurrentUser?.initials || 'U'}</Text>
            )}
          </View>
          <View style={styles.currentUserInfo}>
            <Text style={[styles.currentUserName, { color: colors.text }]} numberOfLines={1}>{syncedCurrentUser?.name || 'You'} (You)</Text>
            <Text style={[styles.currentUserMeta, { color: colors.textSecondary }]}>{syncedCurrentUser?.streak || 0} day streak · {syncedCurrentUser?.totalScans || 0} scans</Text>
          </View>
          <View style={styles.currentUserPointsWrap}>
            <Text style={[styles.currentUserPoints, { color: colors.text }]}>{syncedCurrentUser?.points || 0}</Text>
            <Text style={[styles.currentUserPointsLabel, { color: colors.textSecondary }]}>pts</Text>
          </View>
        </View>
      </Card>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  content: { padding: Spacing.lg },
  scopeCard: {
    marginBottom: Spacing.lg,
  },
  scopeHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: Spacing.md,
  },
  scopeTag: {
    ...Typography.captionMedium,
  },
  scopeTitle: {
    ...Typography.h2,
  },
  scopeBadge: {
    paddingHorizontal: Spacing.md,
    paddingVertical: Spacing.sm,
    borderRadius: BorderRadius.full,
  },
  scopeBadgeText: {
    ...Typography.bodyMedium,
    fontWeight: '700',
  },
  scopeToggle: {
    flexDirection: 'row',
    borderRadius: BorderRadius.full,
    padding: 4,
    gap: 4,
  },
  scopeButton: {
    flex: 1,
    borderRadius: BorderRadius.full,
    borderWidth: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: Spacing.md,
  },
  scopeButtonText: {
    ...Typography.bodyMedium,
    fontWeight: '700',
  },
  centerBlock: {
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: Spacing.xxl,
  },
  stateCard: {
    alignItems: 'center',
    marginBottom: Spacing.lg,
  },
  stateTitle: {
    ...Typography.h3,
    marginBottom: Spacing.xs,
  },
  stateSubtitle: {
    ...Typography.body,
    textAlign: 'center',
  },
  podiumLayout: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    marginBottom: Spacing.lg,
    gap: Spacing.sm,
  },
  podiumSide: {
    flex: 1,
  },
  podiumCenter: {
    flex: 1.18,
  },
  podiumCard: {
    alignItems: 'center',
    paddingVertical: Spacing.md,
  },
  podiumFirst: {
    paddingVertical: Spacing.lg,
    shadowColor: '#10b981',
    shadowOpacity: 0.24,
    shadowRadius: 16,
    shadowOffset: { width: 0, height: 8 },
    elevation: 8,
  },
  placeLabel: {
    ...Typography.h2,
    marginBottom: Spacing.xs,
  },
  podiumAvatar: {
    width: 56,
    height: 56,
    borderRadius: 28,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: Spacing.xs,
    overflow: 'hidden',
  },
  podiumAvatarImage: {
    width: '100%',
    height: '100%',
    resizeMode: 'cover',
  },
  podiumInitials: {
    ...Typography.h3,
    fontWeight: '700',
  },
  podiumName: {
    ...Typography.bodyMedium,
    textAlign: 'center',
  },
  podiumPoints: {
    ...Typography.caption,
  },
  podiumMeta: {
    ...Typography.caption,
    textAlign: 'center',
  },
  listCard: {
    paddingVertical: Spacing.xs,
    marginBottom: Spacing.lg,
  },
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.md,
    paddingVertical: Spacing.sm,
  },
  rank: {
    ...Typography.bodyMedium,
    width: 28,
    textAlign: 'center',
  },
  avatar: {
    width: 38,
    height: 38,
    borderRadius: 19,
    alignItems: 'center',
    justifyContent: 'center',
    overflow: 'hidden',
  },
  avatarImage: {
    width: '100%',
    height: '100%',
    resizeMode: 'cover',
  },
  initials: {
    ...Typography.captionMedium,
    fontWeight: '700',
  },
  userInfo: {
    flex: 1,
    gap: 2,
  },
  userName: {
    ...Typography.bodyMedium,
    fontWeight: '600',
  },
  userMeta: {
    ...Typography.caption,
  },
  points: {
    ...Typography.bodyMedium,
    fontWeight: '700',
    minWidth: 46,
    textAlign: 'right',
  },
  separator: {
    height: 1,
  },
  currentUserCard: {
    borderWidth: 1,
    borderRadius: BorderRadius.lg,
    borderColor: '#10b981',
  },
  currentUserTag: {
    ...Typography.captionMedium,
    marginBottom: Spacing.sm,
  },
  currentUserRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.md,
  },
  rankPill: {
    minWidth: 56,
    minHeight: 36,
    borderRadius: BorderRadius.full,
    alignItems: 'center',
    justifyContent: 'center',
  },
  rankPillText: {
    ...Typography.h3,
    fontWeight: '800',
  },
  currentUserAvatar: {
    width: 52,
    height: 52,
    borderRadius: 26,
    alignItems: 'center',
    justifyContent: 'center',
    overflow: 'hidden',
  },
  currentUserAvatarImage: {
    width: '100%',
    height: '100%',
    resizeMode: 'cover',
  },
  currentUserInitials: {
    ...Typography.h3,
    color: '#fff',
  },
  currentUserInfo: {
    flex: 1,
  },
  currentUserName: {
    ...Typography.h3,
    fontSize: 20,
    marginBottom: 2,
  },
  currentUserMeta: {
    ...Typography.body,
  },
  currentUserPointsWrap: {
    alignItems: 'flex-end',
  },
  currentUserPoints: {
    ...Typography.h2,
  },
  currentUserPointsLabel: {
    ...Typography.body,
  },
});
