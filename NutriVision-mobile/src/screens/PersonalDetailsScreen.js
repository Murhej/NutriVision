import React, { useMemo, useState } from 'react';
import {
  KeyboardAvoidingView,
  Modal,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Button } from '../components';
import { BorderRadius, Shadow, Spacing, Typography } from '../theme';
import { useTheme } from '../theme/ThemeContext';
import { COUNTRIES } from '../data/countries';

const H_PADDING = 18;

const GENDERS = ['Male', 'Female', 'Other'];
const HEIGHT_UNITS = ['cm', 'meter', 'ft/in'];
const WEIGHT_UNITS = ['kg', 'lb'];

const CM_MIN = 50;
const CM_MAX = 250;
const LB_MIN = 50;
const LB_MAX = 1000;

function roundNumber(value, digits = 1) {
  if (!Number.isFinite(value)) return '';
  const rounded = Number(value.toFixed(digits));
  return Number.isInteger(rounded) ? String(rounded) : String(rounded);
}

function parsePositiveNumber(value) {
  const normalized = value.replace(',', '.').trim();
  if (!normalized) return null;
  const parsed = Number(normalized);
  if (!Number.isFinite(parsed) || parsed <= 0) return null;
  return parsed;
}

function parseFeetAndInches(value) {
  const normalized = value.trim();
  if (!normalized) return null;

  const compact = normalized.replace(/\s+/g, '');
  const ftInMatch = compact.match(/^(\d+)(?:'|ft)(\d{1,2})?(?:"|in)?$/i);
  if (ftInMatch) {
    const feet = Number(ftInMatch[1]);
    const inches = ftInMatch[2] ? Number(ftInMatch[2]) : 0;
    if (inches >= 12) return null;
    return feet * 30.48 + inches * 2.54;
  }

  const parts = compact.split(/[:/,-]/).filter(Boolean);
  if (parts.length === 2) {
    const feet = Number(parts[0]);
    const inches = Number(parts[1]);
    if (Number.isFinite(feet) && Number.isFinite(inches) && inches < 12) {
      return feet * 30.48 + inches * 2.54;
    }
  }

  return null;
}

function heightDisplayToCm(value, unit) {
  if (!value.trim()) return null;
  if (unit === 'cm') return parsePositiveNumber(value);
  if (unit === 'meter') {
    const meters = parsePositiveNumber(value);
    return meters ? meters * 100 : null;
  }
  return parseFeetAndInches(value);
}

function cmToHeightDisplay(cmValue, unit) {
  if (!Number.isFinite(cmValue) || cmValue <= 0) return '';
  if (unit === 'cm') return roundNumber(cmValue, 0);
  if (unit === 'meter') return roundNumber(cmValue / 100, 2);

  const totalInches = cmValue / 2.54;
  const feet = Math.floor(totalInches / 12);
  const inches = Math.round(totalInches - feet * 12);
  if (inches === 12) {
    return `${feet + 1}'0`;
  }
  return `${feet}'${inches}`;
}

function weightDisplayToLb(value, unit) {
  if (!value.trim()) return null;
  const parsed = parsePositiveNumber(value);
  if (!parsed) return null;
  return unit === 'kg' ? parsed * 2.2046226218 : parsed;
}

function lbToWeightDisplay(lbValue, unit) {
  if (!Number.isFinite(lbValue) || lbValue <= 0) return '';
  return unit === 'kg' ? roundNumber(lbValue / 2.2046226218, 1) : roundNumber(lbValue, 1);
}

function getTrimmedValue(value) {
  return value.trim();
}

export default function PersonalDetailsScreen({ navigation }) {
  const { colors } = useTheme();
  const insets = useSafeAreaInsets();
  const [firstName, setFirstName] = useState('');
  const [lastName, setLastName] = useState('');
  const [age, setAge] = useState('');
  const [gender, setGender] = useState('');
  const [heightUnit, setHeightUnit] = useState('cm');
  const [height, setHeight] = useState('');
  const [weightUnit, setWeightUnit] = useState('kg');
  const [weight, setWeight] = useState('');
  const [targetWeight, setTargetWeight] = useState('');
  const [country, setCountry] = useState('');
  const [countryQuery, setCountryQuery] = useState('');
  const [isCountryModalVisible, setIsCountryModalVisible] = useState(false);
  const [touched, setTouched] = useState({});

  const setTouchedField = (field) => {
    setTouched((prev) => (prev[field] ? prev : { ...prev, [field]: true }));
  };

  const ageNumber = getTrimmedValue(age) ? Number(age) : null;
  const heightCm = heightDisplayToCm(height, heightUnit);
  const weightLb = weightDisplayToLb(weight, weightUnit);
  const targetWeightLb = weightDisplayToLb(targetWeight, weightUnit);

  const fieldErrors = useMemo(() => {
    const errors = {};

    if (!getTrimmedValue(firstName)) {
      errors.firstName = 'Please enter your first name.';
    }

    if (!getTrimmedValue(lastName)) {
      errors.lastName = 'Please enter your last name.';
    }

    if (!getTrimmedValue(age)) {
      errors.age = 'Please enter your age.';
    } else if (!Number.isInteger(ageNumber) || ageNumber < 12 || ageNumber > 100) {
      errors.age = 'Age must be between 12 and 100.';
    }

    if (!gender) {
      errors.gender = 'Please select one gender option.';
    }

    if (!getTrimmedValue(height)) {
      errors.height = 'Please enter your height.';
    } else if (!heightCm || heightCm < CM_MIN || heightCm > CM_MAX) {
      errors.height = 'Height must be between 50 cm and 250 cm.';
    }

    if (!getTrimmedValue(weight)) {
      errors.weight = 'Please enter your weight.';
    } else if (!weightLb || weightLb < LB_MIN || weightLb > LB_MAX) {
      errors.weight = 'Weight must be between 50 lb and 1000 lb.';
    }

    if (getTrimmedValue(targetWeight) && (!targetWeightLb || targetWeightLb < LB_MIN || targetWeightLb > LB_MAX)) {
      errors.targetWeight = 'Target weight must be between 50 lb and 1000 lb.';
    }

    if (!getTrimmedValue(country)) {
      errors.country = 'Please select your country.';
    }

    return errors;
  }, [age, ageNumber, country, firstName, gender, height, heightCm, lastName, targetWeight, targetWeightLb, weight, weightLb]);

  const filteredCountries = useMemo(() => {
    const query = countryQuery.trim().toLowerCase();
    if (!query) return COUNTRIES;
    return COUNTRIES.filter((item) => item.toLowerCase().includes(query));
  }, [countryQuery]);

  const isFormValid = Object.keys(fieldErrors).length === 0;

  const handleHeightUnitChange = (nextUnit) => {
    if (nextUnit === heightUnit) return;
    const currentCm = heightDisplayToCm(height, heightUnit);
    setHeightUnit(nextUnit);
    if (currentCm) {
      setHeight(cmToHeightDisplay(currentCm, nextUnit));
    }
  };

  const handleWeightUnitChange = (nextUnit) => {
    if (nextUnit === weightUnit) return;
    const currentWeightLb = weightDisplayToLb(weight, weightUnit);
    const currentTargetLb = weightDisplayToLb(targetWeight, weightUnit);

    setWeightUnit(nextUnit);

    if (currentWeightLb) {
      setWeight(lbToWeightDisplay(currentWeightLb, nextUnit));
    }

    if (currentTargetLb) {
      setTargetWeight(lbToWeightDisplay(currentTargetLb, nextUnit));
    }
  };

  const showAllErrors = () => {
    setTouched({
      firstName: true,
      lastName: true,
      age: true,
      gender: true,
      height: true,
      weight: true,
      targetWeight: true,
      country: true,
    });
  };

  const handleContinue = () => {
    if (!isFormValid) {
      showAllErrors();
      return;
    }

    navigation.navigate('Goal', {
      onboardingData: {
        personalInfo: {
          firstName: getTrimmedValue(firstName),
          lastName: getTrimmedValue(lastName),
          fullName: `${getTrimmedValue(firstName)} ${getTrimmedValue(lastName)}`,
          age: getTrimmedValue(age),
          gender,
          height: `${cmToHeightDisplay(heightCm, heightUnit)} ${heightUnit}`,
          heightCm: roundNumber(heightCm, 1),
          heightValue: cmToHeightDisplay(heightCm, heightUnit),
          heightUnit,
          weight: `${lbToWeightDisplay(weightLb, weightUnit)} ${weightUnit}`,
          weightLb: roundNumber(weightLb, 1),
          weightValue: lbToWeightDisplay(weightLb, weightUnit),
          weightUnit,
          targetWeight: getTrimmedValue(targetWeight) ? `${lbToWeightDisplay(targetWeightLb, weightUnit)} ${weightUnit}` : '',
          targetWeightLb: targetWeightLb ? roundNumber(targetWeightLb, 1) : '',
          targetWeightValue: targetWeightLb ? lbToWeightDisplay(targetWeightLb, weightUnit) : '',
          country: getTrimmedValue(country),
        },
      },
    });
  };

  const renderError = (field) => {
    if (!touched[field] || !fieldErrors[field]) return null;
    return <Text style={[styles.errorText, { color: colors.danger }]}>{fieldErrors[field]}</Text>;
  };

  return (
    <KeyboardAvoidingView
      style={[styles.container, { backgroundColor: colors.background }]}
      behavior={Platform.OS === 'ios' ? 'padding' : undefined}
    >
      <View
        style={[
          styles.stickyTop,
          {
            backgroundColor: colors.background,
            paddingTop: insets.top + 8,
            borderBottomColor: colors.borderLight,
          },
        ]}
      >
        <TouchableOpacity
          style={[styles.backButton, { backgroundColor: colors.surface, borderColor: colors.border }]}
          onPress={() => navigation.goBack()}
          activeOpacity={0.8}
        >
          <Text style={[styles.backText, { color: colors.textSecondary }]}>{'<'}</Text>
        </TouchableOpacity>

        <View style={[styles.progressTrack, { backgroundColor: colors.borderLight }]}>
          <View style={[styles.progressFill, { backgroundColor: colors.primary }]} />
        </View>
      </View>
      <ScrollView
        style={styles.scrollView}
        contentContainerStyle={[styles.content, { paddingBottom: 140 + insets.bottom }]}
        showsVerticalScrollIndicator={false}
        keyboardShouldPersistTaps="handled"
      >
        <Text style={[styles.title, { color: colors.text }]}>Tell us about yourself</Text>
        <Text style={[styles.subtitle, { color: colors.textSecondary }]}>Required fields are marked with * and must be valid before you continue.</Text>

        <View style={[styles.sectionCard, { backgroundColor: colors.surface, borderColor: colors.border, shadowColor: colors.shadowColor, shadowOpacity: colors.shadowOpacity }]}> 
          <Text style={[styles.sectionTitle, { color: colors.text }]}>Basic details</Text>

          <Text style={[styles.label, { color: colors.textSecondary }]}>Name *</Text>
          <View style={styles.row}>
            <View style={styles.halfColumn}>
              <TextInput
                style={[
                  styles.input,
                  {
                    borderColor: touched.firstName && fieldErrors.firstName ? colors.danger : colors.border,
                    color: colors.text,
                    backgroundColor: colors.surfaceSecondary,
                  },
                ]}
                placeholder="First name"
                placeholderTextColor={colors.textTertiary}
                value={firstName}
                onChangeText={setFirstName}
                onBlur={() => setTouchedField('firstName')}
              />
              {renderError('firstName')}
            </View>

            <View style={styles.halfColumn}>
              <TextInput
                style={[
                  styles.input,
                  {
                    borderColor: touched.lastName && fieldErrors.lastName ? colors.danger : colors.border,
                    color: colors.text,
                    backgroundColor: colors.surfaceSecondary,
                  },
                ]}
                placeholder="Last name"
                placeholderTextColor={colors.textTertiary}
                value={lastName}
                onChangeText={setLastName}
                onBlur={() => setTouchedField('lastName')}
              />
              {renderError('lastName')}
            </View>
          </View>

          <View style={styles.row}>
            <View style={styles.ageColumn}>
              <Text style={[styles.label, { color: colors.textSecondary }]}>Age *</Text>
              <TextInput
                style={[
                  styles.input,
                  {
                    borderColor: touched.age && fieldErrors.age ? colors.danger : colors.border,
                    color: colors.text,
                    backgroundColor: colors.surfaceSecondary,
                  },
                ]}
                placeholder="18"
                placeholderTextColor={colors.textTertiary}
                value={age}
                onChangeText={setAge}
                onBlur={() => setTouchedField('age')}
                keyboardType="number-pad"
              />
              {renderError('age')}
            </View>

            <View style={styles.genderColumn}>
              <Text style={[styles.label, { color: colors.textSecondary }]}>Gender *</Text>
              <View style={styles.genderRow}>
                {GENDERS.map((item) => {
                  const isSelected = gender === item;
                  return (
                    <TouchableOpacity
                      key={item}
                      style={[
                        styles.genderChip,
                        {
                          borderColor: isSelected ? colors.primary : touched.gender && fieldErrors.gender ? colors.danger : colors.border,
                          backgroundColor: isSelected ? colors.primarySoft : colors.surfaceSecondary,
                        },
                      ]}
                      onPress={() => {
                        setGender(item);
                        setTouchedField('gender');
                      }}
                      activeOpacity={0.85}
                    >
                      <Text style={[styles.optionText, { color: isSelected ? colors.primaryDark : colors.textSecondary }]}>{item}</Text>
                    </TouchableOpacity>
                  );
                })}
              </View>
              {renderError('gender')}
            </View>
          </View>
        </View>

        <View style={[styles.sectionCard, { backgroundColor: colors.surface, borderColor: colors.border, shadowColor: colors.shadowColor, shadowOpacity: colors.shadowOpacity }]}> 
          <Text style={[styles.sectionTitle, { color: colors.text }]}>Body metrics</Text>

          <Text style={[styles.label, { color: colors.textSecondary }]}>Height *</Text>
          <View style={styles.unitToggleRow}>
            {HEIGHT_UNITS.map((item) => {
              const isSelected = heightUnit === item;
              return (
                <TouchableOpacity
                  key={item}
                  style={[
                    styles.unitChip,
                    {
                      borderColor: isSelected ? colors.primary : colors.border,
                      backgroundColor: isSelected ? colors.primarySoft : colors.surfaceSecondary,
                    },
                  ]}
                  onPress={() => handleHeightUnitChange(item)}
                  activeOpacity={0.85}
                >
                  <Text style={[styles.optionText, { color: isSelected ? colors.primaryDark : colors.textSecondary }]}>{item}</Text>
                </TouchableOpacity>
              );
            })}
          </View>
          <TextInput
            style={[
              styles.input,
              {
                borderColor: touched.height && fieldErrors.height ? colors.danger : colors.border,
                color: colors.text,
                backgroundColor: colors.surfaceSecondary,
              },
            ]}
            placeholder={heightUnit === 'ft/in' ? "5'7" : heightUnit === 'meter' ? '1.70' : '170'}
            placeholderTextColor={colors.textTertiary}
            value={height}
            onChangeText={setHeight}
            onBlur={() => setTouchedField('height')}
            keyboardType="decimal-pad"
          />
          <Text style={[styles.helperText, { color: colors.textTertiary }]}>Accepted range: 50 cm to 250 cm.</Text>
          {renderError('height')}

          <Text style={[styles.label, { color: colors.textSecondary }]}>Weight *</Text>
          <View style={styles.unitToggleRow}>
            {WEIGHT_UNITS.map((item) => {
              const isSelected = weightUnit === item;
              return (
                <TouchableOpacity
                  key={item}
                  style={[
                    styles.weightUnitChip,
                    {
                      borderColor: isSelected ? colors.primary : colors.border,
                      backgroundColor: isSelected ? colors.primarySoft : colors.surfaceSecondary,
                    },
                  ]}
                  onPress={() => handleWeightUnitChange(item)}
                  activeOpacity={0.85}
                >
                  <Text style={[styles.optionText, { color: isSelected ? colors.primaryDark : colors.textSecondary }]}>{item}</Text>
                </TouchableOpacity>
              );
            })}
          </View>
          <TextInput
            style={[
              styles.input,
              {
                borderColor: touched.weight && fieldErrors.weight ? colors.danger : colors.border,
                color: colors.text,
                backgroundColor: colors.surfaceSecondary,
              },
            ]}
            placeholder={weightUnit === 'kg' ? '70' : '154'}
            placeholderTextColor={colors.textTertiary}
            value={weight}
            onChangeText={setWeight}
            onBlur={() => setTouchedField('weight')}
            keyboardType="decimal-pad"
          />
          <Text style={[styles.helperText, { color: colors.textTertiary }]}>Accepted range: 50 lb to 1000 lb.</Text>
          {renderError('weight')}

          <Text style={[styles.label, { color: colors.textSecondary }]}>Target Weight</Text>
          <TextInput
            style={[
              styles.input,
              {
                borderColor: touched.targetWeight && fieldErrors.targetWeight ? colors.danger : colors.border,
                color: colors.text,
                backgroundColor: colors.surfaceSecondary,
              },
            ]}
            placeholder={weightUnit === 'kg' ? '65' : '143'}
            placeholderTextColor={colors.textTertiary}
            value={targetWeight}
            onChangeText={setTargetWeight}
            onBlur={() => setTouchedField('targetWeight')}
            keyboardType="decimal-pad"
          />
          <Text style={[styles.helperText, { color: colors.textTertiary }]}>Optional. If entered, it uses the same unit conversion and validation as weight.</Text>
          {renderError('targetWeight')}
        </View>

        <View style={[styles.sectionCard, { backgroundColor: colors.surface, borderColor: colors.border, shadowColor: colors.shadowColor, shadowOpacity: colors.shadowOpacity }]}> 
          <Text style={[styles.sectionTitle, { color: colors.text }]}>Location</Text>
          <Text style={[styles.label, { color: colors.textSecondary }]}>Country *</Text>
          <Pressable
            style={[
              styles.selector,
              {
                borderColor: touched.country && fieldErrors.country ? colors.danger : colors.border,
                backgroundColor: colors.surfaceSecondary,
              },
            ]}
            onPress={() => {
              setTouchedField('country');
              setCountryQuery(country);
              setIsCountryModalVisible(true);
            }}
          >
            <View>
              <Text style={[styles.selectorLabel, { color: colors.textTertiary }]}>Select your country</Text>
              <Text style={[styles.selectorValue, { color: country ? colors.text : colors.textTertiary }]}>{country || 'Search or browse the country list'}</Text>
            </View>
            <Text style={[styles.selectorArrow, { color: colors.textTertiary }]}>⌄</Text>
          </Pressable>
          {renderError('country')}
        </View>
      </ScrollView>

      <View
        style={[
          styles.stickyBottom,
          {
            paddingBottom: insets.bottom + Spacing.md,
            backgroundColor: colors.background,
            borderTopColor: colors.borderLight,
          },
        ]}
      >
        <Button title="Continue" size="lg" onPress={handleContinue} disabled={!isFormValid} style={styles.cta} />
      </View>

      <Modal visible={isCountryModalVisible} animationType="slide" transparent onRequestClose={() => setIsCountryModalVisible(false)}>
        <View style={styles.modalOverlay}>
          <Pressable style={styles.modalBackdrop} onPress={() => setIsCountryModalVisible(false)} />
          <View style={[styles.modalCard, { backgroundColor: colors.surface, paddingBottom: insets.bottom + Spacing.md }]}> 
            <View style={styles.modalHeader}>
              <Text style={[styles.modalTitle, { color: colors.text }]}>Choose your country</Text>
              <TouchableOpacity onPress={() => setIsCountryModalVisible(false)} activeOpacity={0.8}>
                <Text style={[styles.modalClose, { color: colors.textSecondary }]}>Close</Text>
              </TouchableOpacity>
            </View>

            <TextInput
              style={[styles.input, styles.searchInput, { borderColor: colors.border, color: colors.text, backgroundColor: colors.surfaceSecondary }]}
              placeholder="Search countries"
              placeholderTextColor={colors.textTertiary}
              value={countryQuery}
              onChangeText={setCountryQuery}
              autoFocus
            />

            <ScrollView style={styles.countryList} keyboardShouldPersistTaps="handled" showsVerticalScrollIndicator={false}>
              {filteredCountries.length > 0 ? (
                filteredCountries.map((item) => {
                  const isSelected = country === item;
                  return (
                    <TouchableOpacity
                      key={item}
                      style={[
                        styles.countryOption,
                        {
                          borderColor: isSelected ? colors.primary : colors.borderLight,
                          backgroundColor: isSelected ? colors.primarySoft : colors.surface,
                        },
                      ]}
                      onPress={() => {
                        setCountry(item);
                        setTouchedField('country');
                        setCountryQuery(item);
                        setIsCountryModalVisible(false);
                      }}
                      activeOpacity={0.85}
                    >
                      <Text style={[styles.countryOptionText, { color: isSelected ? colors.primaryDark : colors.text }]}>{item}</Text>
                    </TouchableOpacity>
                  );
                })
              ) : (
                <View style={styles.emptyCountryState}>
                  <Text style={[styles.helperText, { color: colors.textSecondary }]}>No countries matched your search.</Text>
                </View>
              )}
            </ScrollView>
          </View>
        </View>
      </Modal>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  stickyTop: {
    paddingHorizontal: H_PADDING,
    paddingBottom: 10,
    borderBottomWidth: 1,
  },
  scrollView: {
    flex: 1,
  },
  content: {
    paddingHorizontal: H_PADDING,
    paddingTop: 18,
    gap: 12,
  },
  backButton: {
    width: 44,
    height: 44,
    borderRadius: 22,
    borderWidth: 1,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 10,
    ...Shadow.sm,
  },
  backText: {
    ...Typography.h2,
    lineHeight: 20,
  },
  progressTrack: {
    height: 6,
    borderRadius: BorderRadius.full,
    overflow: 'hidden',
  },
  progressFill: {
    width: '18%',
    height: '100%',
    borderRadius: BorderRadius.full,
  },
  title: {
    ...Typography.hero,
    marginBottom: 6,
    fontSize: 24,
  },
  subtitle: {
    ...Typography.body,
    fontSize: 15,
    lineHeight: 22,
  },
  sectionCard: {
    borderWidth: 1,
    borderRadius: BorderRadius.lg,
    padding: 16,
    ...Shadow.md,
  },
  sectionTitle: {
    ...Typography.h3,
    marginBottom: Spacing.sm,
  },
  label: {
    ...Typography.bodyMedium,
    marginBottom: 6,
    marginTop: 8,
  },
  helperText: {
    ...Typography.caption,
    marginTop: 4,
  },
  input: {
    borderWidth: 1,
    borderRadius: BorderRadius.lg,
    paddingHorizontal: Spacing.lg,
    paddingVertical: 14,
    fontSize: 15,
  },
  row: {
    flexDirection: 'row',
    gap: Spacing.md,
    alignItems: 'flex-start',
  },
  halfColumn: {
    flex: 1,
  },
  ageColumn: {
    flex: 0.42,
  },
  genderColumn: {
    flex: 0.58,
  },
  genderRow: {
    flexDirection: 'row',
    gap: 8,
    flexWrap: 'wrap',
  },
  genderChip: {
    minWidth: 84,
    borderWidth: 1,
    borderRadius: BorderRadius.lg,
    paddingVertical: 10,
    paddingHorizontal: 10,
    alignItems: 'center',
    justifyContent: 'center',
  },
  unitChip: {
    minWidth: 96,
    borderWidth: 1,
    borderRadius: BorderRadius.lg,
    paddingVertical: 14,
    alignItems: 'center',
    justifyContent: 'center',
  },
  weightUnitChip: {
    minWidth: 84,
    borderWidth: 1,
    borderRadius: BorderRadius.lg,
    paddingVertical: 14,
    alignItems: 'center',
    justifyContent: 'center',
  },
  unitToggleRow: {
    flexDirection: 'row',
    gap: Spacing.sm,
    flexWrap: 'wrap',
    marginBottom: Spacing.sm,
  },
  radioOuter: {
    width: 18,
    height: 18,
    borderRadius: 9,
    borderWidth: 2,
    alignItems: 'center',
    justifyContent: 'center',
  },
  radioInner: {
    width: 8,
    height: 8,
    borderRadius: 4,
  },
  optionText: {
    ...Typography.bodyMedium,
    fontSize: 14,
  },
  errorText: {
    ...Typography.captionMedium,
    marginTop: Spacing.sm,
  },
  selector: {
    borderWidth: 1,
    borderRadius: BorderRadius.lg,
    paddingHorizontal: Spacing.lg,
    paddingVertical: 14,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: Spacing.md,
  },
  selectorLabel: {
    ...Typography.caption,
    marginBottom: 2,
  },
  selectorValue: {
    ...Typography.body,
  },
  selectorArrow: {
    fontSize: 18,
  },
  stickyBottom: {
    borderTopWidth: 1,
    paddingHorizontal: H_PADDING,
    paddingTop: 10,
  },
  cta: {
    width: '100%',
  },
  modalOverlay: {
    flex: 1,
    justifyContent: 'flex-end',
  },
  modalBackdrop: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(15, 23, 42, 0.32)',
  },
  modalCard: {
    borderTopLeftRadius: BorderRadius.xxl,
    borderTopRightRadius: BorderRadius.xxl,
    paddingHorizontal: Spacing.xxl,
    paddingTop: Spacing.xl,
    maxHeight: '78%',
  },
  modalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: Spacing.lg,
  },
  modalTitle: {
    ...Typography.h2,
  },
  modalClose: {
    ...Typography.bodyMedium,
  },
  searchInput: {
    marginBottom: Spacing.lg,
  },
  countryList: {
    flexGrow: 0,
  },
  countryOption: {
    borderWidth: 1,
    borderRadius: BorderRadius.lg,
    paddingHorizontal: Spacing.lg,
    paddingVertical: Spacing.md,
    marginBottom: Spacing.sm,
  },
  countryOptionText: {
    ...Typography.body,
  },
  emptyCountryState: {
    alignItems: 'center',
    paddingVertical: Spacing.xxxl,
  },
});
