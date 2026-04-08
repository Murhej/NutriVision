import React, { useState } from 'react';
import { View, Text, TextInput, StyleSheet, ScrollView } from 'react-native';
import { Button } from '../components';

export default function PersonalDetailsScreen({ navigation }) {
  const [name, setName] = useState('');
  const [age, setAge] = useState('');
  const [gender, setGender] = useState('');
  const [country, setCountry] = useState('');

  const handleContinue = () => {
    if (!name || !age || !gender || !country) {
      alert('Please fill all fields');
      return;
    }

    navigation.navigate('Goal'); // temporary (we will change later)
  };

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.title}>Enter Your Details</Text>

      <TextInput
        placeholder="Name"
        style={styles.input}
        value={name}
        onChangeText={setName}
      />

      <TextInput
        placeholder="Age"
        style={styles.input}
        keyboardType="numeric"
        value={age}
        onChangeText={setAge}
      />

      <TextInput
        placeholder="Gender"
        style={styles.input}
        value={gender}
        onChangeText={setGender}
      />

      <TextInput
        placeholder="Country"
        style={styles.input}
        value={country}
        onChangeText={setCountry}
      />

      <Button title="Continue" onPress={handleContinue} />
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flexGrow: 1,
    justifyContent: 'center',
    padding: 20,
  },
  title: {
    fontSize: 24,
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