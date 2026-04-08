import React, { useState } from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import { Button } from '../components';

export default function ActivityLevelScreen({ navigation }) {
  const [selected, setSelected] = useState('');

  const levels = [
    'Mostly sitting (desk job)',
    'Moderately active',
    'Physically active job',
  ];

  const handleContinue = () => {
    if (!selected) {
      alert('Please select your activity level');
      return;
    }

    navigation.navigate('Exercise'); // temporary
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>What is your daily activity level?</Text>

      {levels.map((item, index) => (
        <TouchableOpacity
          key={index}
          style={[
            styles.option,
            selected === item && styles.selected
          ]}
          onPress={() => setSelected(item)}
        >
          <Text>{item}</Text>
        </TouchableOpacity>
      ))}

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
  option: {
    padding: 15,
    borderWidth: 1,
    borderColor: '#ccc',
    borderRadius: 10,
    marginBottom: 10,
  },
  selected: {
    backgroundColor: '#e0f2fe',
    borderColor: '#0284c7',
  },
});