import React, { useState } from 'react';
import { View, Text, StyleSheet, ScrollView, TouchableOpacity } from 'react-native';
import { useTheme } from '../theme/ThemeContext';
import { Typography, Spacing, BorderRadius, Shadow } from '../theme';
import { Card } from '../components';
import { FEED_ARTICLES, FEED_CATEGORIES } from '../data/mockData';

export default function FeedScreen() {
  const { colors } = useTheme();
  const [activeCategory, setActiveCategory] = useState('All');

  const filtered =
    activeCategory === 'All'
      ? FEED_ARTICLES
      : FEED_ARTICLES.filter((a) => a.category === activeCategory);

  return (
    <ScrollView
      style={[styles.container, { backgroundColor: colors.background }]}
      contentContainerStyle={styles.content}
      showsVerticalScrollIndicator={false}
    >
      {/* Category Pills */}
      <ScrollView
        horizontal
        showsHorizontalScrollIndicator={false}
        contentContainerStyle={styles.pills}
      >
        {FEED_CATEGORIES.map((cat) => (
          <TouchableOpacity
            key={cat}
            style={[
              styles.pill,
              {
                backgroundColor: activeCategory === cat ? colors.primary : colors.surface,
                borderColor: activeCategory === cat ? colors.primary : colors.border,
              },
            ]}
            onPress={() => setActiveCategory(cat)}
            activeOpacity={0.7}
          >
            <Text
              style={[
                styles.pillText,
                { color: activeCategory === cat ? '#fff' : colors.textSecondary },
              ]}
            >
              {cat}
            </Text>
          </TouchableOpacity>
        ))}
      </ScrollView>

      {/* Articles */}
      <View style={styles.articles}>
        {filtered.map((article) => (
          <TouchableOpacity key={article.id} activeOpacity={0.85}>
            <Card style={styles.articleCard}>
              <View style={styles.articleRow}>
                <View
                  style={[
                    styles.articleEmoji,
                    { backgroundColor: article.color || colors.primarySoft },
                  ]}
                >
                  <Text style={styles.emojiText}>{article.emoji}</Text>
                </View>
                <View style={styles.articleContent}>
                  <View style={styles.articleMeta}>
                    <Text style={[styles.categoryTag, { color: colors.primary }]}>
                      {article.category}
                    </Text>
                    <Text style={[styles.readTime, { color: colors.textTertiary }]}>
                      {article.readTime}
                    </Text>
                  </View>
                  <Text style={[styles.articleTitle, { color: colors.text }]} numberOfLines={2}>
                    {article.title}
                  </Text>
                  <Text style={[styles.articleExcerpt, { color: colors.textSecondary }]} numberOfLines={2}>
                    {article.excerpt}
                  </Text>
                </View>
              </View>
            </Card>
          </TouchableOpacity>
        ))}
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  content: { padding: Spacing.lg, paddingBottom: Spacing.xxxl * 2 },
  pills: {
    flexDirection: 'row',
    gap: Spacing.sm,
    marginBottom: Spacing.xl,
    paddingRight: Spacing.lg,
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
  articles: {
    gap: Spacing.md,
  },
  articleCard: {
    padding: Spacing.md,
  },
  articleRow: {
    flexDirection: 'row',
    gap: Spacing.md,
  },
  articleEmoji: {
    width: 64,
    height: 64,
    borderRadius: BorderRadius.md,
    alignItems: 'center',
    justifyContent: 'center',
  },
  emojiText: {
    fontSize: 28,
  },
  articleContent: {
    flex: 1,
    gap: Spacing.xs,
  },
  articleMeta: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.sm,
  },
  categoryTag: {
    ...Typography.small,
  },
  readTime: {
    ...Typography.small,
  },
  articleTitle: {
    ...Typography.bodyMedium,
    fontWeight: '700',
  },
  articleExcerpt: {
    ...Typography.caption,
    lineHeight: 18,
  },
});
