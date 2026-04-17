import React, { useCallback, useEffect, useState } from 'react';
import { ActivityIndicator, RefreshControl, ScrollView, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useTheme } from '../theme/ThemeContext';
import { Typography, Spacing, BorderRadius } from '../theme';
import { Card } from '../components';
import { useAuth } from '../context/AuthContext';

export default function FeedScreen() {
  const { colors } = useTheme();
  const insets = useSafeAreaInsets();
  const { fetchFeed } = useAuth();

  const [activeTopic, setActiveTopic] = useState('All');
  const [topics, setTopics] = useState(['All']);
  const [articles, setArticles] = useState([]);
  const [hasEnoughData, setHasEnoughData] = useState(false);
  const [preparationMessage, setPreparationMessage] = useState('');
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState('');

  const loadFeed = useCallback(async (topic = activeTopic, isRefresh = false) => {
    try {
      if (isRefresh) {
        setRefreshing(true);
      } else {
        setLoading(true);
      }
      setError('');
      const data = await fetchFeed(topic);
      const nextTopics = Array.isArray(data?.topics) && data.topics.length ? data.topics : ['All'];
      setTopics(nextTopics);
      setArticles(Array.isArray(data?.articles) ? data.articles : []);
      setHasEnoughData(Boolean(data?.hasEnoughData));
      setPreparationMessage(data?.preparationMessage || 'Personalized content will appear after more activity.');

      if (!nextTopics.includes(topic)) {
        setActiveTopic('All');
      }
    } catch (e) {
      if (e?.status === 404) {
        setTopics(['All']);
        setArticles([]);
        setHasEnoughData(false);
        setPreparationMessage('We are preparing your personalized feed. Keep logging meals and goals.');
        setError('');
      } else {
        setError(e?.message || 'Unable to load feed.');
      }
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, [activeTopic, fetchFeed]);

  useEffect(() => {
    loadFeed('All');
  }, [loadFeed]);

  const onSelectTopic = async (topic) => {
    setActiveTopic(topic);
    await loadFeed(topic);
  };

  return (
    <ScrollView
      style={[styles.container, { backgroundColor: colors.background }]}
      contentContainerStyle={[styles.content, { paddingTop: insets.top + 8, paddingBottom: insets.bottom + 72 }]}
      showsVerticalScrollIndicator={false}
      refreshControl={<RefreshControl refreshing={refreshing} onRefresh={() => loadFeed(activeTopic, true)} tintColor={colors.primary} />}
    >
      <Card style={[styles.headerCard, { borderColor: colors.primaryLight, backgroundColor: colors.primarySoft }]}> 
        <Text style={[styles.headerTag, { color: colors.primary }]}>FOR YOU</Text>
        <Text style={[styles.headerTitle, { color: colors.text }]}>Feed</Text>
        <Text style={[styles.headerSubtitle, { color: colors.textSecondary }]}>Bite-sized nutrition reads tailored to your goals and habits.</Text>
      </Card>

      <Card style={styles.statsCard}>
        <View style={styles.statsRow}>
          <Text style={[styles.statsValue, { color: colors.text }]}>{articles.length}</Text>
          <Text style={[styles.statsLabel, { color: colors.textSecondary }]}>articles in {activeTopic}</Text>
        </View>
        <Text style={[styles.statsHint, { color: colors.textSecondary }]}>
          {hasEnoughData ? 'Content updates as your logs and profile evolve.' : 'Keep logging meals and goals to unlock personalized reads.'}
        </Text>
      </Card>

      <Text style={[styles.sectionTitle, { color: colors.text }]}>Topics</Text>
      <Text style={[styles.sectionSubtitle, { color: colors.textSecondary }]}>Tap to filter</Text>

      <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.pills}>
        {topics.map((topic) => (
          <TouchableOpacity
            key={topic}
            style={[
              styles.pill,
              {
                backgroundColor: activeTopic === topic ? colors.primarySoft : colors.surface,
                borderColor: activeTopic === topic ? colors.primary : colors.border,
              },
            ]}
            onPress={() => onSelectTopic(topic)}
            activeOpacity={0.75}
          >
            <Text style={[styles.pillText, { color: activeTopic === topic ? colors.primary : colors.textSecondary }]}>{topic}</Text>
          </TouchableOpacity>
        ))}
      </ScrollView>

      {loading ? (
        <View style={styles.centerBlock}>
          <ActivityIndicator size="large" color={colors.primary} />
        </View>
      ) : error ? (
        <Card style={styles.emptyCard}>
          <Text style={[styles.emptyTitle, { color: colors.text }]}>Unable to load feed</Text>
          <Text style={[styles.emptySubtitle, { color: colors.textSecondary }]}>{error}</Text>
        </Card>
      ) : articles.length === 0 ? (
        <Card style={styles.emptyCard}>
          <Text style={[styles.emptyTitle, { color: colors.text }]}>We are preparing your personalized feed</Text>
          <Text style={[styles.emptySubtitle, { color: colors.textSecondary }]}>{preparationMessage}</Text>
        </Card>
      ) : (
        <View style={styles.articles}>
          {articles.map((article) => (
            <Card key={article.id} style={styles.articleCard}>
              <View style={styles.articleMeta}>
                <Text style={[styles.articleTopic, { color: colors.primary }]}>{article.topic}</Text>
                <Text style={[styles.articleTime, { color: colors.textTertiary }]}>{article.readTime}</Text>
              </View>
              <Text style={[styles.articleTitle, { color: colors.text }]}>{article.title}</Text>
              <Text style={[styles.articleExcerpt, { color: colors.textSecondary }]}>{article.excerpt}</Text>
              <View style={[styles.reasonTag, { backgroundColor: colors.surfaceSecondary }]}> 
                <Text style={[styles.reasonText, { color: colors.textSecondary }]}>{article.reason}</Text>
              </View>
            </Card>
          ))}
        </View>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  content: { padding: Spacing.lg },
  headerCard: {
    borderWidth: 1,
    marginBottom: Spacing.md,
  },
  headerTag: {
    ...Typography.captionMedium,
    marginBottom: 2,
  },
  headerTitle: {
    ...Typography.h1,
    marginBottom: Spacing.xs,
  },
  headerSubtitle: {
    ...Typography.body,
    maxWidth: 320,
  },
  statsCard: {
    marginBottom: Spacing.lg,
  },
  statsRow: {
    flexDirection: 'row',
    alignItems: 'baseline',
    gap: Spacing.sm,
    marginBottom: Spacing.sm,
  },
  statsValue: {
    ...Typography.h1,
    fontSize: 26,
  },
  statsLabel: {
    ...Typography.body,
  },
  statsHint: {
    ...Typography.body,
  },
  sectionTitle: {
    ...Typography.h2,
  },
  sectionSubtitle: {
    ...Typography.body,
    marginBottom: Spacing.sm,
  },
  pills: {
    gap: Spacing.sm,
    paddingBottom: Spacing.md,
    marginBottom: Spacing.md,
  },
  pill: {
    paddingHorizontal: Spacing.lg,
    paddingVertical: Spacing.sm,
    borderRadius: BorderRadius.full,
    borderWidth: 1,
  },
  pillText: {
    ...Typography.captionMedium,
  },
  centerBlock: {
    paddingVertical: Spacing.xxl,
    alignItems: 'center',
    justifyContent: 'center',
  },
  emptyCard: {
    alignItems: 'center',
    paddingVertical: Spacing.xxl,
    marginTop: Spacing.lg,
  },
  emptyTitle: {
    ...Typography.h3,
    textAlign: 'center',
    marginBottom: Spacing.sm,
  },
  emptySubtitle: {
    ...Typography.body,
    textAlign: 'center',
    maxWidth: 300,
  },
  articles: {
    gap: Spacing.md,
  },
  articleCard: {
    gap: Spacing.xs,
  },
  articleMeta: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 2,
  },
  articleTopic: {
    ...Typography.captionMedium,
  },
  articleTime: {
    ...Typography.caption,
  },
  articleTitle: {
    ...Typography.bodyMedium,
    fontWeight: '700',
  },
  articleExcerpt: {
    ...Typography.caption,
    lineHeight: 18,
  },
  reasonTag: {
    alignSelf: 'flex-start',
    borderRadius: BorderRadius.full,
    paddingHorizontal: Spacing.md,
    paddingVertical: 5,
    marginTop: Spacing.xs,
  },
  reasonText: {
    ...Typography.small,
  },
});
