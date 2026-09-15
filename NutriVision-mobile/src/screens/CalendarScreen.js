import React, { useState, useMemo, useCallback, useEffect } from 'react';
import { View, Text, StyleSheet, ScrollView, TouchableOpacity, Alert, Modal } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useFocusEffect } from '@react-navigation/native';
import { useTheme } from '../theme/ThemeContext';
import { Typography, Spacing, BorderRadius, Shadow } from '../theme';
import { Card } from '../components';
import { ChevronLeft, ChevronRight, Trash2 } from 'lucide-react-native';
import { useMeals } from '../context/MealContext';
import { useAuth } from '../context/AuthContext';
import { apiRequest } from '../api/client';

const DAY_LABELS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];

export default function CalendarScreen() {
  const { colors } = useTheme();
  const insets = useSafeAreaInsets();
  const { getMealsForDate, getSummaryForDate, activeDateKeys, loadMeals, upsertMeals, removeMeal, personalizedTargets } = useMeals();
  const { token } = useAuth();
  const today = new Date();
  const [viewMonth, setViewMonth] = useState(today.getMonth());
  const [viewYear, setViewYear] = useState(today.getFullYear());
  const [selectedDate, setSelectedDate] = useState(
    `${today.getFullYear()}-${String(today.getMonth() + 1).padStart(2, '0')}-${String(today.getDate()).padStart(2, '0')}`
  );
  const [selectedMeal, setSelectedMeal] = useState(null);
  const [isDeleting, setIsDeleting] = useState(false);

  // Reload meals from backend every time this screen comes into focus
  // Limit to 30 meals for faster loading (covers ~30-40 days of history)
  useFocusEffect(
    useCallback(() => {
      if (!token) return;
      apiRequest('/map/logs?limit=30', { token })
        .then((data) => {
          if (data?.entries) loadMeals(data.entries);
        })
        .catch(() => {});
    }, [token, loadMeals]),
  );

  useEffect(() => {
    if (!token || !selectedDate) return;
    apiRequest(`/map/logs?date=${encodeURIComponent(selectedDate)}&limit=100`, { token })
      .then((data) => {
        if (data?.entries) upsertMeals(data.entries);
      })
      .catch(() => {});
  }, [selectedDate, token, upsertMeals]);

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
      cells.push({
        day: d,
        key: dateStr,
        dateStr,
      });
    }
    return cells;
  }, [viewMonth, viewYear]);

  const monthName = new Date(viewYear, viewMonth).toLocaleString('default', { month: 'long', year: 'numeric' });

  const dailyCalorieTarget = Number(personalizedTargets?.targets?.calories) || 2200;
  const selectedSummary = getSummaryForDate(selectedDate, dailyCalorieTarget);
  const selectedMeals = getMealsForDate(selectedDate);

  const getStatusColor = (status) => {
    switch (status) {
      case 'on-track': return colors.success;
      case 'low': return colors.warning || colors.primary;
      case 'over': return colors.danger;
      default: return 'transparent';
    }
  };

  const isToday = (dateStr) => {
    const todayStr = `${today.getFullYear()}-${String(today.getMonth() + 1).padStart(2, '0')}-${String(today.getDate()).padStart(2, '0')}`;
    return dateStr === todayStr;
  };

  const handleDeleteMeal = async (meal) => {
    setIsDeleting(true);
    try {
      await removeMeal(meal.id, token);
      setSelectedMeal(null);
    } catch (err) {
      Alert.alert('Delete Failed', err?.message || 'Could not delete meal. Please try again.');
    } finally {
      setIsDeleting(false);
    }
  };

  const confirmDelete = (meal) => {
    Alert.alert(
      'Delete Meal',
      `Remove "${meal.name}" (${meal.calories} kcal)?`,
      [
        { text: 'Cancel', style: 'cancel' },
        { text: 'Delete', style: 'destructive', onPress: () => handleDeleteMeal(meal) },
      ],
    );
  };

  return (
    <View style={[styles.rootContainer, { backgroundColor: colors.background }]}>
    <ScrollView
      style={styles.container}
      contentContainerStyle={[
        styles.content,
        {
          paddingTop: insets.top + Spacing.md,
          paddingBottom: insets.bottom + 72,
        },
      ]}
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
                      {activeDateKeys.has(cell.dateStr) && cell.dateStr !== selectedDate && (() => {
                        const s = getSummaryForDate(cell.dateStr, dailyCalorieTarget);
                        return s ? <View style={[styles.dot, { backgroundColor: getStatusColor(s.status) }]} /> : null;
                      })()}
                </>
              ) : null}
            </TouchableOpacity>
          ))}
        </View>
      </Card>

      {/* Selected Day Summary */}
      {selectedSummary ? (
        <Card style={styles.summaryCard}>
          <Text style={[styles.summaryTitle, { color: colors.text }]}>Daily Summary</Text>
          <View style={styles.summaryRow}>
            <View style={styles.summaryItem}>
              <Text style={[styles.summaryValue, { color: colors.calories }]}>
                {selectedSummary.consumed}
              </Text>
              <Text style={[styles.summaryLabel, { color: colors.textSecondary }]}>Consumed</Text>
            </View>
            <View style={[styles.divider, { backgroundColor: colors.border }]} />
            <View style={styles.summaryItem}>
              <Text style={[styles.summaryValue, { color: colors.text }]}>{selectedSummary.goal}</Text>
              <Text style={[styles.summaryLabel, { color: colors.textSecondary }]}>Goal</Text>
            </View>
            <View style={[styles.divider, { backgroundColor: colors.border }]} />
            <View style={styles.summaryItem}>
              <Text
                style={[
                  styles.summaryValue,
                  {
                    color:
                      selectedSummary.status === 'on-track'
                        ? colors.success
                        : selectedSummary.status === 'over'
                          ? colors.danger
                          : colors.warning || colors.primary,
                  },
                ]}
              >
                {selectedSummary.status === 'on-track' ? '✓' : selectedSummary.status === 'over' ? '↑' : '•'}
              </Text>
              <Text style={[styles.summaryLabel, { color: colors.textSecondary }]}>Status</Text>
            </View>
          </View>
        </Card>
      ) : (
        <Card style={styles.summaryCard}>
          <Text style={[styles.summaryTitle, { color: colors.textSecondary }]}>
            No meals logged for this day.
          </Text>
        </Card>
      )}

      {/* Meal list for selected day */}
      {selectedMeals.length > 0 && (
        <View style={styles.mealSection}>
          <Text style={[styles.mealSectionTitle, { color: colors.text }]}>Meals</Text>
          {selectedMeals.map((meal) => (
            <TouchableOpacity
              key={meal.id}
              onPress={() => setSelectedMeal(meal)}
              activeOpacity={0.75}
            >
              <Card style={styles.mealRow}>
                <Text style={[styles.mealName, { color: colors.text }]}>{meal.name}</Text>
                <View style={styles.mealMeta}>
                  <Text style={[styles.mealTime, { color: colors.textSecondary }]}>
                    {meal.timestamp.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })}
                  </Text>
                  <Text style={[styles.mealCal, { color: colors.text }]}>{meal.calories} kcal</Text>
                </View>
              </Card>
            </TouchableOpacity>
          ))}
        </View>
      )}
    </ScrollView>

    {/* Meal action sheet */}
    <Modal
      visible={Boolean(selectedMeal)}
      transparent
      animationType="slide"
      onRequestClose={() => setSelectedMeal(null)}
    >
      <TouchableOpacity
        style={[styles.overlay, { backgroundColor: '#00000060' }]}
        activeOpacity={1}
        onPress={() => setSelectedMeal(null)}
      >
        <View
          style={[
            styles.actionSheet,
            { backgroundColor: colors.surface, paddingBottom: insets.bottom + Spacing.lg },
          ]}
          onStartShouldSetResponder={() => true}
        >
          {selectedMeal && (
            <>
              <View style={[styles.sheetHandle, { backgroundColor: colors.border }]} />
              <Text style={[styles.sheetTitle, { color: colors.text }]}>{selectedMeal.name}</Text>
              <Text style={[styles.sheetSub, { color: colors.textSecondary }]}>
                {selectedMeal.timestamp.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })} •{' '}
                {selectedMeal.calories} kcal
              </Text>

              <View style={styles.sheetMacros}>
                {[
                  { label: 'Protein', value: selectedMeal.protein, unit: 'g', color: colors.protein },
                  { label: 'Carbs', value: selectedMeal.carbs, unit: 'g', color: colors.carbs },
                  { label: 'Fat', value: selectedMeal.fat, unit: 'g', color: colors.fat },
                ].map((m) => (
                  <View key={m.label} style={styles.sheetMacroItem}>
                    <Text style={[styles.sheetMacroValue, { color: m.color }]}>{m.value}</Text>
                    <Text style={[styles.sheetMacroUnit, { color: colors.textTertiary }]}>{m.unit}</Text>
                    <Text style={[styles.sheetMacroLabel, { color: colors.textSecondary }]}>{m.label}</Text>
                  </View>
                ))}
              </View>

              <TouchableOpacity
                style={[styles.deleteBtn, { backgroundColor: colors.error + '18', borderColor: colors.error }]}
                onPress={() => confirmDelete(selectedMeal)}
                disabled={isDeleting}
                activeOpacity={0.8}
              >
                <Trash2 size={18} color={colors.error} />
                <Text style={[styles.deleteBtnText, { color: colors.error }]}>
                  {isDeleting ? 'Deleting…' : 'Delete Meal'}
                </Text>
              </TouchableOpacity>

              <TouchableOpacity
                style={[styles.cancelBtn, { borderColor: colors.border }]}
                onPress={() => setSelectedMeal(null)}
                activeOpacity={0.8}
              >
                <Text style={[styles.cancelBtnText, { color: colors.text }]}>Cancel</Text>
              </TouchableOpacity>
            </>
          )}
        </View>
      </TouchableOpacity>
    </Modal>
  </View>
  );
}

const styles = StyleSheet.create({
  rootContainer: { flex: 1 },
  container: { flex: 1 },
  content: { paddingHorizontal: Spacing.lg },
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
    marginBottom: Spacing.lg,
  },
  mealSectionTitle: {
    ...Typography.h3,
    marginBottom: Spacing.sm,
  },
  mealRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: Spacing.xs,
    paddingVertical: Spacing.sm,
  },
  mealName: {
    ...Typography.body,
    flex: 1,
  },
  mealMeta: {
    alignItems: 'flex-end',
    gap: 2,
  },
  mealTime: {
    ...Typography.caption,
  },
  mealCal: {
    ...Typography.captionMedium,
    fontWeight: '600',
  },

  // Action sheet / modal
  overlay: {
    flex: 1,
    justifyContent: 'flex-end',
  },
  actionSheet: {
    borderTopLeftRadius: BorderRadius.xxxl,
    borderTopRightRadius: BorderRadius.xxxl,
    padding: Spacing.lg,
    gap: Spacing.md,
    ...Shadow.lg,
  },
  sheetHandle: {
    width: 40,
    height: 4,
    borderRadius: 2,
    alignSelf: 'center',
    marginBottom: Spacing.sm,
  },
  sheetTitle: {
    ...Typography.h2,
    fontWeight: '700',
    textTransform: 'capitalize',
  },
  sheetSub: {
    ...Typography.caption,
    marginTop: -Spacing.xs,
  },
  sheetMacros: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: Spacing.md,
    borderTopWidth: 1,
    borderBottomWidth: 1,
    borderColor: '#e0e0e022',
    marginVertical: Spacing.sm,
  },
  sheetMacroItem: { alignItems: 'center', flex: 1 },
  sheetMacroValue: { ...Typography.h3, fontWeight: '700' },
  sheetMacroUnit: { ...Typography.small },
  sheetMacroLabel: { ...Typography.caption, marginTop: 2 },
  deleteBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: Spacing.sm,
    paddingVertical: Spacing.md,
    borderRadius: BorderRadius.lg,
    borderWidth: 1.5,
  },
  deleteBtnText: { ...Typography.bodyMedium, fontWeight: '700' },
  cancelBtn: {
    paddingVertical: Spacing.md,
    borderRadius: BorderRadius.lg,
    borderWidth: 1,
    alignItems: 'center',
  },
  cancelBtnText: { ...Typography.bodyMedium, fontWeight: '600' },
});
