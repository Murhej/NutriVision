import React, { useState } from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import { Button } from '../components';

export default function ExerciseTypeScreen({ navigation }) {
  const [selected, setSelected] = useState([]);

  const types = [
    'Home Workout',
    'Gym',
    'Outdoor Activities',
    'Sports',
  ];

  const toggle = (item) => {
    if (selected.includes(item)) {
      setSelected(selected.filter(i => i !== item));
    } else {
      setSelected([...selected, item]);
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>How do you prefer to exercise?</Text>

      {types.map((item, index) => (
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

      <Button
        title="Continue"
        onPress={() => navigation.navigate('Login')}
      />

      <TouchableOpacity onPress={() => navigation.navigate('Login')}>
        <Text style={styles.skip}>Skip</Text>
      </TouchableOpacity>
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
    backgroundColor: '#ede9fe',
    borderColor: '#7c3aed',
  },
  skip: {
    marginTop: 15,
    textAlign: 'center',
    color: 'gray',
  },
});