import React, { useState } from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import { Button } from '../components';

export default function PreferencesScreen({ navigation }) {
  const [selected, setSelected] = useState([]);

  const preferences = [
    'Vegetarian',
    'High Protein',
    'Low Carb',
    'Vegan',
    'Balanced Diet',
  ];

  const toggle = (item) => {
    if (selected.includes(item)) {
      setSelected(selected.filter(i => i !== item));
    } else {
      setSelected([...selected, item]);
    }
  };

  const handleContinue = () => {
    if (selected.length === 0) {
      alert('Please select at least one preference');
      return;
    }

    navigation.navigate('Reaction'); // temporary
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Select your daily preferences</Text>

      {preferences.map((item, index) => (
        <TouchableOpacity
          key={index}
          style={[
            styles.option,
            selected.includes(item) && styles.selected
          ]}
          onPress={() => toggle(item)}
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
    backgroundColor: '#dbeafe',
    borderColor: '#2563eb',
  },
});