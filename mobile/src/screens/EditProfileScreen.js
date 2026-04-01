import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, TextInput, Alert, ScrollView, Switch } from 'react-native';
import { useTheme } from '../theme/ThemeContext';
import { Typography, Spacing, BorderRadius } from '../theme';
import { Button } from '../components';
import { apiClient } from '../api/client';
import { ArrowLeft } from 'lucide-react-native';
import { TouchableOpacity } from 'react-native';

export default function EditProfileScreen({ navigation }) {
  const { colors } = useTheme();
  
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  
  const [name, setName] = useState('');
  const [goal, setGoal] = useState('Maintain Weight');
  const [calories, setCalories] = useState('2000');
  const [protein, setProtein] = useState('120');
  const [carbs, setCarbs] = useState('250');
  const [fat, setFat] = useState('65');
  const [metric, setMetric] = useState(true);

  useEffect(() => {
    const fetchCurrent = async () => {
      try {
        const res = await apiClient.get('/api/mobile/profile');
        const p = res.profile;
        if (p) {
          setName(p.name);
          setGoal(p.goal);
          setCalories(p.dailyCalorieGoal + "");
          setProtein(p.proteinGoal + "");
          setCarbs(p.carbsGoal + "");
          setFat(p.fatGoal + "");
          setMetric(p.unitSystem === "Metric");
        }
      } catch (e) {
        Alert.alert('Error', 'Could not load your profile');
      } finally {
        setLoading(false);
      }
    };
    fetchCurrent();
  }, []);

  const handleSave = async () => {
    setSaving(true);
    try {
      await apiClient.post('/api/mobile/profile', {
        name,
        goal,
        dailyCalorieGoal: parseInt(calories) || 2000,
        proteinGoal: parseInt(protein) || 120,
        carbsGoal: parseInt(carbs) || 250,
        fatGoal: parseInt(fat) || 65,
        unitSystem: metric ? "Metric" : "Imperial"
      });
      Alert.alert('Success', 'Profile updated!', [
        { text: 'OK', onPress: () => navigation.goBack() }
      ]);
    } catch (e) {
      Alert.alert('Error', 'Failed to save changes');
    } finally {
      setSaving(false);
    }
  };

  if (loading) return null;

  return (
    <View style={{flex: 1, backgroundColor: colors.background}}>
      <View style={[styles.header, { backgroundColor: colors.surface, borderBottomColor: colors.borderLight }]}>
        <TouchableOpacity onPress={() => navigation.goBack()} style={styles.backButton}>
          <ArrowLeft size={24} color={colors.text} />
        </TouchableOpacity>
        <Text style={[styles.headerTitle, { color: colors.text }]}>Edit Profile</Text>
        <View style={{ width: 24 }} />
      </View>
      <ScrollView contentContainerStyle={styles.content}>
        
        <Text style={[styles.sectionTitle, { color: colors.text }]}>Personal Details</Text>
        
        <Text style={[styles.label, { color: colors.textSecondary }]}>Name</Text>
        <TextInput
          style={[styles.input, { borderColor: colors.border, color: colors.text }]}
          value={name}
          onChangeText={setName}
        />

        <Text style={[styles.label, { color: colors.textSecondary }]}>Primary Goal</Text>
        <TextInput
          style={[styles.input, { borderColor: colors.border, color: colors.text }]}
          value={goal}
          onChangeText={setGoal}
          placeholder="e.g. Lose Weight, Build Muscle"
          placeholderTextColor={colors.textTertiary}
        />

        <Text style={[styles.sectionTitle, { color: colors.text, marginTop: Spacing.lg }]}>Daily Nutrition Limits</Text>

        <Text style={[styles.label, { color: colors.textSecondary }]}>Daily Calories (kcal)</Text>
        <TextInput
          style={[styles.input, { borderColor: colors.border, color: colors.text }]}
          value={calories}
          onChangeText={setCalories}
          keyboardType="numeric"
        />

        <Text style={[styles.label, { color: colors.textSecondary }]}>Protein Goal (g)</Text>
        <TextInput
          style={[styles.input, { borderColor: colors.border, color: colors.text }]}
          value={protein}
          onChangeText={setProtein}
          keyboardType="numeric"
        />

        <Text style={[styles.label, { color: colors.textSecondary }]}>Carbs Goal (g)</Text>
        <TextInput
          style={[styles.input, { borderColor: colors.border, color: colors.text }]}
          value={carbs}
          onChangeText={setCarbs}
          keyboardType="numeric"
        />

        <Text style={[styles.label, { color: colors.textSecondary }]}>Fat Goal (g)</Text>
        <TextInput
          style={[styles.input, { borderColor: colors.border, color: colors.text }]}
          value={fat}
          onChangeText={setFat}
          keyboardType="numeric"
        />
        
        <Text style={[styles.sectionTitle, { color: colors.text, marginTop: Spacing.lg }]}>Preferences</Text>
        
        <View style={styles.switchRow}>
          <Text style={[styles.label, { color: colors.text, marginBottom: 0 }]}>
            Use Metric System
          </Text>
          <Switch
            value={metric}
            onValueChange={setMetric}
            trackColor={{ false: colors.border, true: colors.primaryLight }}
            thumbColor={metric ? colors.primary : colors.textTertiary}
          />
        </View>

        <Button
          title={saving ? "Saving..." : "Save Changes"}
          size="lg"
          onPress={handleSave}
          disabled={saving}
          style={{ marginTop: Spacing.xxxl }}
        />
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: Spacing.lg,
    paddingTop: 60,
    paddingBottom: Spacing.md,
    borderBottomWidth: 1,
  },
  backButton: {
    padding: Spacing.xs,
  },
  headerTitle: {
    ...Typography.h3,
  },
  content: {
    padding: Spacing.xl,
    paddingBottom: Spacing.xxxl * 2,
  },
  sectionTitle: {
    ...Typography.h4,
    marginBottom: Spacing.md,
  },
  label: { ...Typography.captionMedium, marginBottom: Spacing.xs },
  input: {
    borderWidth: 1,
    borderRadius: BorderRadius.md,
    padding: Spacing.md,
    fontSize: 16,
    marginBottom: Spacing.md,
  },
  switchRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: Spacing.sm,
  }
});
