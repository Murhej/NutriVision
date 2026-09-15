import React, { useMemo, useState } from 'react';
import {
  Alert,
  LayoutAnimation,
  Platform,
  RefreshControl,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  UIManager,
  View,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useTheme } from '../theme/ThemeContext';
import { Typography, Spacing, BorderRadius } from '../theme';
import { Card } from '../components';
import { useAuth } from '../context/AuthContext';
import { useMeals } from '../context/MealContext';
import { FIXED_ACHIEVEMENT_DEFS, RANDOM_ACHIEVEMENT_POOL, seededShuffle } from '../data/achievementsCatalog';

const HUB_TABS = ['featured', 'active_random', 'milestones'];

export default function AchievementsScreen() {
  const { colors } = useTheme();
  const insets = useSafeAreaInsets();
  const { achievements, profile, upsertAchievement } = useAuth();
  const { getAchievementMetrics, getXpProgression } = useMeals();
  const [expandedId, setExpandedId] = useState(null);
  const [activeTab, setActiveTab] = useState('featured');

  React.useEffect(() => {
    if (Platform.OS === 'android' && UIManager.setLayoutAnimationEnabledExperimental) {
      UIManager.setLayoutAnimationEnabledExperimental(true);
    }
  }, []);

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
    const todayKey = today.toISOString().slice(0, 10);
    const dateSeed = Number(`${today.getFullYear()}${today.getMonth() + 1}${today.getDate()}`);
    const userSeed = String(profile?.id || profile?.email || 'guest').split('').reduce((sum, ch) => sum + ch.charCodeAt(0), 0);
    const completedTodayBaseIds = new Set(
      (achievements || [])
        .filter((item) => {
          const status = String(item?.status || '').toLowerCase();
          const isDone = status === 'completed' || status === 'claimed';
          return isDone && String(item?.id || '').endsWith(`_${todayKey}`);
        })
        .map((item) => String(item.id).replace(`_${todayKey}`, '')),
    );
    const shuffledPool = seededShuffle(RANDOM_ACHIEVEMENT_POOL, dateSeed + userSeed);
    const eligiblePool = shuffledPool.filter((item) => !completedTodayBaseIds.has(item.id));
    const selected = eligiblePool.slice(0, 3);
    return selected.map((item) => ({ ...item, id: `${item.id}_${today.toISOString().slice(0, 10)}`, type: 'random' }));
  }, [achievements, profile]);

  const evaluateAchievement = React.useCallback((def) => {
    const rawProgress = Number(achievementMetrics?.[def.metric] || 0);
    const progress = Math.max(0, rawProgress);
    const target = Number(def.target || 1);
    const ratio = target > 0 ? progress / target : 0;
    const persisted = achievementStateMap.get(def.id);
    const completedByProgress = target > 0 ? progress >= target : false;
    const persistedStatus = String(persisted?.status || '').toLowerCase();

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
      persisted,
      completedAt: persisted?.completed_at || persisted?.completedAt || (completedByProgress ? new Date().toISOString() : null),
      xpReward: Number(def.xpReward || 0),
      bonusXp: Number(def.bonusXp || 0),
    };
  }, [achievementMetrics, achievementStateMap]);

  const fixedAchievements = useMemo(() => FIXED_ACHIEVEMENT_DEFS.map(evaluateAchievement), [evaluateAchievement]);
  const randomAchievements = useMemo(() => activeRandomDefs.map(evaluateAchievement), [activeRandomDefs, evaluateAchievement]);
  const allAchievements = useMemo(() => [...fixedAchievements, ...randomAchievements], [fixedAchievements, randomAchievements]);

  const featuredAchievements = useMemo(() => {
    const merged = [...fixedAchievements, ...randomAchievements]
      .filter((item) => item.status !== 'claimed')
      .map((item) => {
        const baseValue = Number(item.xpReward || 0);
        const progressRatio = Math.max(0, Math.min(item.ratio || 0, 1));
        const isInProgress = item.status === 'in_progress';
        const isCompleted = item.status === 'completed';
        const behavioralBoost = isInProgress ? 80 : 0;
        const completionBoost = isCompleted ? 120 : 0;
        const recommendationScore = baseValue + progressRatio * 100 + behavioralBoost + completionBoost;
        return { ...item, recommendationScore };
      })
      .sort((a, b) => b.recommendationScore - a.recommendationScore);
    return merged.slice(0, 6);
  }, [fixedAchievements, randomAchievements]);

  const completedCount = useMemo(
    () => allAchievements.filter((item) => item.status === 'completed' || item.status === 'claimed').length,
    [allAchievements],
  );

  const achievementXpEarned = useMemo(
    () => allAchievements
      .filter((item) => item.status === 'completed' || item.status === 'claimed')
      .reduce((sum, item) => sum + Number(item.xpReward || 0), 0),
    [allAchievements],
  );

  const completedSyncKey = useMemo(
    () => allAchievements
      .filter((item) => item.status === 'completed')
      .map((item) => item.id)
      .sort()
      .join('|'),
    [allAchievements],
  );

  React.useEffect(() => {
    let cancelled = false;

    const persistCompletedAchievements = async () => {
      const pending = allAchievements.filter((item) => {
        if (item.status !== 'completed') return false;
        const persistedStatus = String(item?.persisted?.status || '').toLowerCase();
        return persistedStatus !== 'completed' && persistedStatus !== 'claimed';
      });

      for (const item of pending) {
        if (cancelled) return;
        try {
          await upsertAchievement({
            id: item.id,
            user_id: profile?.id || profile?.user_id || profile?.userId || null,
            title: item.title,
            description: item.description,
            type: item.type,
            category: item.type,
            difficulty: item.difficulty,
            progress: item.progress,
            target: item.target,
            status: 'completed',
            completed_at: item.completedAt || new Date().toISOString(),
            xp_reward: item.xpReward,
          });
        } catch (error) {
          // Skip hard failure so the hub remains responsive if network is unavailable.
        }
      }
    };

    persistCompletedAchievements();

    return () => {
      cancelled = true;
    };
  }, [allAchievements, completedSyncKey, profile, upsertAchievement]);

  const claimAchievement = async (achievement) => {
    if (!achievement) return;
    if (achievement.status === 'claimed') {
      Alert.alert('Already Claimed', 'This reward has already been added.');
      return;
    }
    if (achievement.status !== 'completed') {
      Alert.alert('Not Ready Yet', 'Complete progress first to claim this reward.');
      return;
    }

    try {
      await upsertAchievement({
        id: achievement.id,
        user_id: profile?.id || profile?.user_id || profile?.userId || null,
        title: achievement.title,
        description: achievement.description,
        type: achievement.type,
        category: achievement.type,
        difficulty: achievement.difficulty,
        progress: achievement.progress,
        target: achievement.target,
        status: 'claimed',
        completed_at: achievement.completedAt || new Date().toISOString(),
        claimed_at: new Date().toISOString(),
        xp_reward: achievement.xpReward,
      });
      Alert.alert('Reward Claimed', `+${achievement.xpReward} XP`);
    } catch (error) {
      Alert.alert('Save Failed', 'Could not claim achievement right now.');
    }
  };

  const toggleExpanded = (achievementId) => {
    LayoutAnimation.configureNext({
      duration: 240,
      create: { type: 'easeInEaseOut', property: 'opacity' },
      update: { type: 'easeInEaseOut' },
      delete: { type: 'easeInEaseOut', property: 'opacity' },
    });
    setExpandedId((prev) => (prev === achievementId ? null : achievementId));
  };

  const getTabTitle = (tab) => {
    if (tab === 'featured') return 'Featured';
    if (tab === 'active_random') return 'Active Random Challenges';
    return 'Milestones';
  };

  const getChallengeExpiryText = (achievementId) => {
    const idText = String(achievementId || '');
    const dateSuffix = idText.slice(-10);
    if (!/^\d{4}-\d{2}-\d{2}$/.test(dateSuffix)) return null;

    const end = new Date(`${dateSuffix}T23:59:59`);
    const deltaMs = end.getTime() - Date.now();
    if (Number.isNaN(deltaMs) || deltaMs <= 0) return 'Expires soon';

    const hours = Math.floor(deltaMs / (1000 * 60 * 60));
    const minutes = Math.floor((deltaMs % (1000 * 60 * 60)) / (1000 * 60));
    return `Expires in ${hours}h ${minutes}m`;
  };

  const getProgressUnit = (item) => {
    const metric = String(item?.metric || '');
    if (metric.includes('streak') || metric.includes('days')) return 'days completed';
    if (metric.includes('meal')) return 'meals completed';
    if (metric.includes('water') || metric.includes('hydration')) return 'hydration goals completed';
    return 'completed';
  };

  const getConditions = (item) => {
    const metric = String(item?.metric || '').toLowerCase();
    const type = String(item?.type || '').toLowerCase();
    const target = Number(item?.target || 1);
    let counts = `Complete ${target} required milestones for this achievement.`;
    let resets = 'Progress does not reset.';

    if (metric.includes('streak')) {
      counts = `Maintain your streak until ${target} consecutive days are reached.`;
      resets = 'Missing a qualifying day resets streak progress.';
    } else if (metric.includes('meal')) {
      counts = `Log qualifying meals until you reach ${target}.`;
    } else if (metric.includes('water') || metric.includes('hydration')) {
      counts = `Meet your hydration target on ${target} qualifying day(s).`;
    } else if (metric.includes('macro')) {
      counts = `Hit macro goals on ${target} qualifying day(s).`;
    } else if (metric.includes('calorie')) {
      counts = `Stay within calorie target band for ${target} qualifying day(s).`;
    } else if (metric.includes('vitamin') || metric.includes('mineral')) {
      counts = `Complete micronutrient goals on ${target} qualifying day(s).`;
    }

    if (type === 'random' && !metric.includes('streak')) {
      resets = 'Challenge rotates at daily reset if not completed.';
    }

    return { counts, resets };
  };

  const renderAchievementCard = (item, showExpiry = false) => {
    const statusLabel = item.status === 'claimed' ? 'Completed' : item.status === 'completed' ? 'Completed' : item.status === 'in_progress' ? 'In Progress' : 'Locked';
    const isExpanded = expandedId === item.id;
    const expiryText = showExpiry ? getChallengeExpiryText(item.id) : null;
    const progressValue = Math.min(item.progress, item.target);
    const progressUnit = getProgressUnit(item);
    const { counts, resets } = getConditions(item);
    const totalXpShown = Number(item.xpReward || 0) + Number(item.bonusXp || 0);
    const isCompleted = item.status === 'completed' || item.status === 'claimed';
    const isLocked = item.status === 'locked';
    const statusIcon = isCompleted ? '✓' : isLocked ? '🔒' : '•';
    const cardHighlightStyle = isCompleted
      ? { borderColor: colors.success || '#22c55e', borderWidth: 1, backgroundColor: 'rgba(34,197,94,0.08)' }
      : isLocked
        ? { borderColor: colors.border, borderWidth: 1 }
        : {};

    return (
      <Card key={item.id} style={[styles.itemCard, cardHighlightStyle]}>
        <TouchableOpacity onPress={() => toggleExpanded(item.id)} activeOpacity={0.9}>
          <View style={styles.cardTopRow}>
            <View style={styles.cardTopLeft}>
              <Text style={styles.emoji}>{item.emoji || '🏅'}</Text>
              <Text style={[styles.itemTitle, { color: colors.text }]} numberOfLines={1}>{item.title}</Text>
            </View>
            <View style={styles.cardTopRight}>
              <Text style={[styles.topXpText, { color: colors.primary }]}>+{totalXpShown} XP</Text>
              <Text style={[styles.arrowText, { color: colors.textSecondary }]}>{isExpanded ? '↑' : '↓'}</Text>
            </View>
          </View>

          <View style={styles.collapsedMetaRow}>
            <View style={[styles.statusPill, { backgroundColor: colors.surfaceSecondary }]}> 
              <Text style={[styles.statusText, { color: colors.textSecondary }]}>{statusIcon} {statusLabel}</Text>
            </View>
            <View style={[styles.metaPill, { backgroundColor: colors.primarySoft }]}> 
              <Text style={[styles.metaPillText, { color: colors.primary }]}>{String(item.difficulty || 'easy').toUpperCase()}</Text>
            </View>
          </View>

          <View style={styles.progressRow}>
            <Text style={[styles.progressText, { color: colors.textSecondary }]}>Progress: {progressValue}/{item.target}</Text>
            <Text style={[styles.progressPercent, { color: colors.textSecondary }]}>{Math.round(Math.max(0, Math.min(item.ratio * 100, 100)))}%</Text>
          </View>

          <View style={[styles.progressTrack, { backgroundColor: colors.surfaceSecondary }]}> 
            <View style={[styles.progressFill, { backgroundColor: colors.primary, width: `${Math.max(0, Math.min(item.ratio * 100, 100))}%` }]} />
          </View>

          {isLocked ? (
            <Text style={[styles.lockedHint, { color: colors.textSecondary }]} numberOfLines={1}>Unlock by reaching {item.target} progress.</Text>
          ) : null}

          {expiryText ? <Text style={[styles.expiryText, { color: colors.textSecondary }]}>{expiryText}</Text> : null}
        </TouchableOpacity>

        {isExpanded ? (
          <View style={[styles.dropdownContent, { backgroundColor: colors.surfaceSecondary, borderColor: colors.border }]}> 
            <Text style={[styles.expandedDescription, { color: colors.textSecondary }]}>{item.description}</Text>

            <View style={styles.dropdownRow}>
              <Text style={[styles.dropdownLabel, { color: colors.textSecondary }]}>Progress</Text>
              <Text style={[styles.dropdownValue, { color: colors.text }]}>{progressValue} / {item.target} {progressUnit}</Text>
            </View>
            <View style={styles.dropdownRow}>
              <Text style={[styles.dropdownLabel, { color: colors.textSecondary }]}>Difficulty</Text>
              <Text style={[styles.dropdownValue, { color: colors.text }]}>{String(item.difficulty || 'easy').toUpperCase()}</Text>
            </View>
            <View style={styles.dropdownRow}>
              <Text style={[styles.dropdownLabel, { color: colors.textSecondary }]}>XP Breakdown</Text>
              <Text style={[styles.dropdownValue, { color: colors.text }]}>Base {Number(item.xpReward || 0)} • Bonus {Number(item.bonusXp || 0)}</Text>
            </View>

            <View style={styles.conditionsWrap}>
              <Text style={[styles.conditionsTitle, { color: colors.text }]}>Conditions</Text>
              <Text style={[styles.conditionsText, { color: colors.textSecondary }]}>Counts: {counts}</Text>
              <Text style={[styles.conditionsText, { color: colors.textSecondary }]}>Resets: {resets}</Text>
            </View>

            {isCompleted ? (
              <View style={styles.dropdownRow}>
                <Text style={[styles.dropdownLabel, { color: colors.textSecondary }]}>XP Earned</Text>
                <Text style={[styles.dropdownValue, { color: colors.success || '#22c55e' }]}>+{totalXpShown} XP</Text>
              </View>
            ) : null}

            {item.completedAt ? (
              <View style={styles.dropdownRow}>
                <Text style={[styles.dropdownLabel, { color: colors.textSecondary }]}>Completion Date</Text>
                <Text style={[styles.dropdownValue, { color: colors.text }]}>{new Date(item.completedAt).toLocaleDateString()}</Text>
              </View>
            ) : null}
          </View>
        ) : null}

        {item.status === 'completed' && (
          <TouchableOpacity style={[styles.claimBtn, { backgroundColor: colors.primary }]} onPress={() => claimAchievement(item)}>
            <Text style={styles.claimBtnText}>Claim +{item.xpReward} XP</Text>
          </TouchableOpacity>
        )}
      </Card>
    );
  };

  const tabItems = useMemo(() => {
    if (activeTab === 'featured') return featuredAchievements;
    if (activeTab === 'active_random') return randomAchievements;
    return fixedAchievements;
  }, [activeTab, featuredAchievements, randomAchievements, fixedAchievements]);

  return (
    <ScrollView
      style={[styles.container, { backgroundColor: colors.background }]}
      contentContainerStyle={[styles.content, { paddingTop: insets.top + 8, paddingBottom: insets.bottom + 72 }]}
      refreshControl={<RefreshControl refreshing={false} onRefresh={() => {}} tintColor={colors.primary} />}
      showsVerticalScrollIndicator={false}
    >
      <Card style={styles.summaryCard}>
        <Text style={[styles.summaryTitle, { color: colors.text }]}>Achievements Hub</Text>
        <Text style={[styles.summarySub, { color: colors.textSecondary }]}>Current Level: {xpProgression.level} • Total XP: {Math.round(xpProgression.totalXp)}</Text>
        <View style={styles.summaryStatsRow}>
          <View style={[styles.summaryStatPill, { backgroundColor: colors.surfaceSecondary }]}>
            <Text style={[styles.summaryStatLabel, { color: colors.textSecondary }]}>Achievement XP</Text>
            <Text style={[styles.summaryStatValue, { color: colors.text }]}>{Math.round(achievementXpEarned)}</Text>
          </View>
          <View style={[styles.summaryStatPill, { backgroundColor: colors.surfaceSecondary }]}>
            <Text style={[styles.summaryStatLabel, { color: colors.textSecondary }]}>Completed</Text>
            <Text style={[styles.summaryStatValue, { color: colors.text }]}>{completedCount}</Text>
          </View>
          <View style={[styles.summaryStatPill, { backgroundColor: colors.surfaceSecondary }]}>
            <Text style={[styles.summaryStatLabel, { color: colors.textSecondary }]}>Active Challenges</Text>
            <Text style={[styles.summaryStatValue, { color: colors.text }]}>{randomAchievements.length}</Text>
          </View>
        </View>
        <View style={[styles.levelTrack, { backgroundColor: colors.surfaceSecondary }]}> 
          <View style={[styles.levelFill, { backgroundColor: colors.primary, width: `${Math.max(0, Math.min((xpProgression.xpIntoLevel / 500) * 100, 100))}%` }]} />
        </View>
        <Text style={[styles.levelMeta, { color: colors.textSecondary }]}>Level progress: {Math.round(xpProgression.xpIntoLevel)} / 500 XP</Text>
      </Card>

      <View style={styles.tabRow}>
        {HUB_TABS.map((tab) => {
          const active = activeTab === tab;
          return (
            <TouchableOpacity
              key={tab}
              style={[
                styles.tabPill,
                {
                  backgroundColor: active ? colors.primary : colors.surfaceSecondary,
                  borderColor: active ? colors.primary : colors.border,
                },
              ]}
              onPress={() => setActiveTab(tab)}
              activeOpacity={0.9}
            >
              <Text style={[styles.tabPillText, { color: active ? '#fff' : colors.textSecondary }]}>{getTabTitle(tab)}</Text>
            </TouchableOpacity>
          );
        })}
      </View>

      <View style={styles.tabHeaderWrap}>
        <Text style={[styles.tabHeader, { color: colors.text }]}>{getTabTitle(activeTab)}</Text>
        <Text style={[styles.tabSub, { color: colors.textSecondary }]}>
          {activeTab === 'featured'
            ? 'Recommended goals and high-value achievements based on your current behavior.'
            : activeTab === 'active_random'
              ? 'Rotating challenges that refresh when completed or at daily reset.'
              : 'Long-term progression milestones across scanning, streaks, hydration, and consistency.'}
        </Text>
      </View>

      <View style={styles.list}>
        {tabItems.length === 0 ? (
          <Card style={[styles.emptyCard, { borderColor: colors.border, backgroundColor: colors.surfaceSecondary }]}> 
            <Text style={[styles.emptyTitle, { color: colors.text }]}>No achievements yet</Text>
            <Text style={[styles.emptySub, { color: colors.textSecondary }]}>Start logging meals and hydration to unlock progress.</Text>
          </Card>
        ) : (
          tabItems.map((item) => renderAchievementCard(item, activeTab === 'active_random'))
        )}
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  content: { padding: Spacing.lg },
  summaryCard: { marginBottom: Spacing.md },
  summaryTitle: { ...Typography.h1, marginBottom: 2 },
  summarySub: { ...Typography.body },
  summaryStatsRow: { flexDirection: 'row', gap: Spacing.xs, marginTop: Spacing.md },
  summaryStatPill: {
    flex: 1,
    borderRadius: BorderRadius.md,
    paddingVertical: Spacing.xs,
    paddingHorizontal: Spacing.sm,
  },
  summaryStatLabel: { ...Typography.caption },
  summaryStatValue: { ...Typography.bodyMedium, fontWeight: '700' },
  levelTrack: {
    marginTop: Spacing.sm,
    height: 8,
    borderRadius: BorderRadius.full,
    overflow: 'hidden',
  },
  levelFill: {
    height: '100%',
    borderRadius: BorderRadius.full,
  },
  levelMeta: { ...Typography.caption, marginTop: 4 },
  tabRow: {
    flexDirection: 'row',
    gap: 8,
    flexWrap: 'wrap',
    marginBottom: Spacing.md,
  },
  tabPill: {
    borderWidth: 1,
    borderRadius: BorderRadius.full,
    paddingHorizontal: Spacing.md,
    paddingVertical: 8,
  },
  tabPillText: {
    ...Typography.captionMedium,
    fontWeight: '700',
  },
  tabHeaderWrap: { marginBottom: Spacing.sm },
  tabHeader: { ...Typography.h2 },
  tabSub: { ...Typography.body, marginTop: 2 },
  list: { gap: Spacing.md },
  emptyCard: {
    borderWidth: 1,
    borderRadius: BorderRadius.md,
    padding: Spacing.md,
  },
  emptyTitle: {
    ...Typography.bodyMedium,
    fontWeight: '700',
    marginBottom: 4,
  },
  emptySub: {
    ...Typography.body,
  },
  itemCard: { gap: Spacing.xs, paddingTop: Spacing.md },
  cardTopRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  cardTopLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    flex: 1,
  },
  cardTopRight: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  topXpText: { ...Typography.bodyMedium, fontWeight: '700' },
  arrowText: { ...Typography.h3, lineHeight: 18 },
  emoji: { fontSize: 22 },
  collapsedMetaRow: {
    flexDirection: 'row',
    gap: 8,
    marginTop: Spacing.xs,
    alignItems: 'center',
  },
  statusPill: {
    borderRadius: BorderRadius.full,
    paddingHorizontal: Spacing.md,
    paddingVertical: 6,
  },
  statusText: { ...Typography.captionMedium },
  itemTitle: { ...Typography.h3, flex: 1 },
  metaPill: {
    borderRadius: BorderRadius.full,
    paddingHorizontal: Spacing.md,
    paddingVertical: 6,
  },
  metaPillText: { ...Typography.caption },
  progressRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginTop: Spacing.xs,
  },
  progressText: { ...Typography.caption },
  progressPercent: { ...Typography.captionMedium },
  progressTrack: {
    height: 6,
    borderRadius: BorderRadius.full,
    overflow: 'hidden',
    marginTop: 2,
  },
  progressFill: {
    height: '100%',
    borderRadius: BorderRadius.full,
  },
  lockedHint: {
    ...Typography.caption,
    marginTop: 4,
  },
  expiryText: { ...Typography.caption, marginTop: 4 },
  dropdownContent: {
    marginTop: Spacing.xs,
    borderWidth: 1,
    borderRadius: BorderRadius.md,
    padding: Spacing.sm,
    gap: 6,
  },
  expandedDescription: { ...Typography.body },
  dropdownRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  dropdownLabel: {
    ...Typography.caption,
  },
  dropdownValue: {
    ...Typography.captionMedium,
  },
  conditionsWrap: {
    marginTop: 2,
    gap: 2,
  },
  conditionsTitle: {
    ...Typography.bodyMedium,
    fontWeight: '700',
  },
  conditionsText: {
    ...Typography.caption,
  },
  claimBtn: {
    marginTop: Spacing.sm,
    minHeight: 38,
    borderRadius: BorderRadius.md,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: Spacing.md,
  },
  claimBtnText: {
    ...Typography.bodyMedium,
    color: '#fff',
    fontWeight: '700',
  },
});
