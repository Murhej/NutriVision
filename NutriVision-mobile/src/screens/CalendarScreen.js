import React, { useState, useMemo } from 'react';
import { View, Text, StyleSheet, ScrollView, TouchableOpacity } from 'react-native';
import { useTheme } from '../theme/ThemeContext';
import { Typography, Spacing, BorderRadius, Shadow } from '../theme';
import { Card, MealCard } from '../components';
import { CALENDAR_DATA, TODAYS_MEALS } from '../data/mockData';
import { ChevronLeft, ChevronRight } from 'lucide-react-native';

const DAY_LABELS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];

export default function CalendarScreen() {
  const { colors } = useTheme();
  const today = new Date();
  const [viewMonth, setViewMonth] = useState(today.getMonth());
  const [viewYear, setViewYear] = useState(today.getFullYear());
  const [selectedDate, setSelectedDate] = useState(
    `${today.getFullYear()}-${String(today.getMonth() + 1).padStart(2, '0')}-${String(today.getDate()).padStart(2, '0')}`
  );

  const calendarDays = useMemo(() => {
    const firstDay = new Date(viewYear, viewMonth, 1).getDay();
    const daysInMonth = new Date(viewYear, viewMonth + 1, 0).getDate();
    const cells = [];

    // Blank cells for days before the 1st
    for (let i = 0; i < firstDay; i++) {
      cells.push({ day: null, key: `blank-${i}` });
    }
    // Actual days
    for (let d = 1; d <= daysInMonth; d++) {
      const dateStr = `${viewYear}-${String(viewMonth + 1).padStart(2, '0')}-${String(d).padStart(2, '0')}`;
      const data = CALENDAR_DATA[dateStr];
      cells.push({
        day: d,
        key: dateStr,
        dateStr,
        status: data?.status || 'no-data',
        calories: data?.calories || 0,
        goal: data?.goal || 2000,
      });
    }
    return cells;
  }, [viewMonth, viewYear]);

  const monthName = new Date(viewYear, viewMonth).toLocaleString('default', { month: 'long', year: 'numeric' });

  const selectedData = CALENDAR_DATA[selectedDate];

  const getStatusColor = (status) => {
    switch (status) {
      case 'on-target': return colors.success;
      case 'over': return colors.danger;
      default: return 'transparent';
    }
  };

  const isToday = (dateStr) => {
    const todayStr = `${today.getFullYear()}-${String(today.getMonth() + 1).padStart(2, '0')}-${String(today.getDate()).padStart(2, '0')}`;
    return dateStr === todayStr;
  };

  return (
    <ScrollView
      style={[styles.container, { backgroundColor: colors.background }]}
      contentContainerStyle={styles.content}
      showsVerticalScrollIndicator={false}
    >
      {/* Month Navigation */}
      <View style={styles.monthNav}>
        <TouchableOpacity
          onPress={() => {
            if (viewMonth === 0) {
              setViewMonth(11);
              setViewYear(viewYear - 1);
            } else {
              setViewMonth(viewMonth - 1);
            }
          }}
          style={[styles.navBtn, { backgroundColor: colors.surface, borderColor: colors.border }]}
        >
          <ChevronLeft size={20} color={colors.text} />
        </TouchableOpacity>
        <Text style={[styles.monthTitle, { color: colors.text }]}>{monthName}</Text>
        <TouchableOpacity
          onPress={() => {
            if (viewMonth === 11) {
              setViewMonth(0);
              setViewYear(viewYear + 1);
            } else {
              setViewMonth(viewMonth + 1);
            }
          }}
          style={[styles.navBtn, { backgroundColor: colors.surface, borderColor: colors.border }]}
        >
          <ChevronRight size={20} color={colors.text} />
        </TouchableOpacity>
      </View>

      {/* Day Labels */}
      <View style={styles.dayLabelsRow}>
        {DAY_LABELS.map((label) => (
          <Text key={label} style={[styles.dayLabel, { color: colors.textTertiary }]}>
            {label}
          </Text>
        ))}
      </View>

      {/* Calendar Grid */}
      <Card style={styles.calendarCard}>
        <View style={styles.grid}>
          {calendarDays.map((cell) => (
            <TouchableOpacity
              key={cell.key}
              style={[
                styles.cell,
                cell.dateStr === selectedDate && {
                  backgroundColor: colors.primary,
                  borderRadius: BorderRadius.sm,
                },
              ]}
              onPress={() => cell.dateStr && setSelectedDate(cell.dateStr)}
              disabled={!cell.day}
            >
              {cell.day ? (
                <>
                  <Text
                    style={[
                      styles.cellDay,
                      { color: cell.dateStr === selectedDate ? '#fff' : colors.text },
                      isToday(cell.dateStr) && cell.dateStr !== selectedDate && { color: colors.primary, fontWeight: '800' },
                    ]}
                  >
                    {cell.day}
                  </Text>
                  {cell.status !== 'no-data' && cell.dateStr !== selectedDate && (
                    <View style={[styles.dot, { backgroundColor: getStatusColor(cell.status) }]} />
                  )}
                </>
              ) : null}
            </TouchableOpacity>
          ))}
        </View>
      </Card>

      {/* Selected Day Summary */}
      {selectedData && selectedData.status !== 'no-data' ? (
        <Card style={styles.summaryCard}>
          <Text style={[styles.summaryTitle, { color: colors.text }]}>Daily Summary</Text>
          <View style={styles.summaryRow}>
            <View style={styles.summaryItem}>
              <Text style={[styles.summaryValue, { color: colors.calories }]}>
                {selectedData.calories}
              </Text>
              <Text style={[styles.summaryLabel, { color: colors.textSecondary }]}>Consumed</Text>
            </View>
            <View style={[styles.divider, { backgroundColor: colors.border }]} />
            <View style={styles.summaryItem}>
              <Text style={[styles.summaryValue, { color: colors.text }]}>{selectedData.goal}</Text>
              <Text style={[styles.summaryLabel, { color: colors.textSecondary }]}>Goal</Text>
            </View>
            <View style={[styles.divider, { backgroundColor: colors.border }]} />
            <View style={styles.summaryItem}>
              <Text
                style={[
                  styles.summaryValue,
                  { color: selectedData.status === 'on-target' ? colors.success : colors.danger },
                ]}
              >
                {selectedData.status === 'on-target' ? '✓' : '↑'}
              </Text>
              <Text style={[styles.summaryLabel, { color: colors.textSecondary }]}>Status</Text>
            </View>
          </View>
        </Card>
      ) : (
        <Card style={styles.summaryCard}>
          <Text style={[styles.summaryTitle, { color: colors.textSecondary }]}>
            No data for the selected day
          </Text>
        </Card>
      )}

      {/* Show today's meals if today is selected */}
      {selectedDate === `${today.getFullYear()}-${String(today.getMonth() + 1).padStart(2, '0')}-${String(today.getDate()).padStart(2, '0')}` && (
        <View style={styles.mealSection}>
          <Text style={[styles.mealTitle, { color: colors.text }]}>Meals</Text>
          {TODAYS_MEALS.map((meal) => (
            <MealCard key={meal.id} meal={meal} />
          ))}
        </View>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  content: { padding: Spacing.lg, paddingBottom: Spacing.xxxl * 2 },
  monthNav: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: Spacing.lg,
  },
  navBtn: {
    width: 40,
    height: 40,
    borderRadius: 20,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1,
  },
  monthTitle: { ...Typography.h2 },
  dayLabelsRow: {
    flexDirection: 'row',
    marginBottom: Spacing.sm,
  },
  dayLabel: {
    flex: 1,
    textAlign: 'center',
    ...Typography.small,
  },
  calendarCard: {
    padding: Spacing.sm,
    marginBottom: Spacing.lg,
  },
  grid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
  },
  cell: {
    width: '14.28%',
    aspectRatio: 1,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 2,
  },
  cellDay: {
    ...Typography.bodyMedium,
    fontSize: 14,
  },
  dot: {
    width: 5,
    height: 5,
    borderRadius: 5,
    marginTop: 2,
  },
  summaryCard: {
    marginBottom: Spacing.lg,
  },
  summaryTitle: {
    ...Typography.h3,
    marginBottom: Spacing.md,
  },
  summaryRow: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  summaryItem: {
    flex: 1,
    alignItems: 'center',
    gap: Spacing.xs,
  },
  summaryValue: {
    ...Typography.h2,
  },
  summaryLabel: {
    ...Typography.caption,
  },
  divider: {
    width: 1,
    height: 36,
  },
  mealSection: {
    gap: Spacing.md,
  },
  mealTitle: {
    ...Typography.h3,
  },
});
