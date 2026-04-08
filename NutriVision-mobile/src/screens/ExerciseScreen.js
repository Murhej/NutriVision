import React, { useState } from 'react';
import { View, Text, StyleSheet, TextInput } from 'react-native';
import { Button } from '../components';

export default function ExerciseScreen({ navigation }) {
  const [level, setLevel] = useState('');
  const [hours, setHours] = useState('');

  const handleContinue = () => {
    if (!level || !hours) {
      alert('Please fill all fields');
      return;
    }

    navigation.navigate('ExerciseType');
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Tell us about your exercise habits</Text>

      <TextInput
        placeholder="Activity level (e.g. beginner, moderate)"
        style={styles.input}
        value={level}
        onChangeText={setLevel}
      />

      <TextInput
        placeholder="Hours per week"
        style={styles.input}
        keyboardType="numeric"
        value={hours}
        onChangeText={setHours}
      />

      <Button title="Continue" onPress={handleContinue} />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
    justifyContent: 'center',
  },
  title: {
    fontSize: 22,
    marginBottom: 20,
    textAlign: 'center',
  },
  input: {
    borderWidth: 1,
    borderColor: '#ccc',
    padding: 12,
    borderRadius: 8,
    marginBottom: 15,
  },
});