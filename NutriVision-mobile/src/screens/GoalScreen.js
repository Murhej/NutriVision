import React, { useState } from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import { Button } from '../components';

export default function GoalScreen({ navigation }) {
  const [selectedGoals, setSelectedGoals] = useState([]);

  const goals = [
    'Lose Weight',
    'Build Muscle',
    'Stay Healthy',
    'Improve Diet',
  ];

  const toggleGoal = (goal) => {
    if (selectedGoals.includes(goal)) {
      setSelectedGoals(selectedGoals.filter(g => g !== goal));
    } else {
      setSelectedGoals([...selectedGoals, goal]);
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>What is your main goal?</Text>

      {goals.map((goal, index) => (
        <TouchableOpacity
          key={index}
          style={[
            styles.option,
            selectedGoals.includes(goal) && styles.selected
          ]}
          onPress={() => toggleGoal(goal)}
        >
          <Text style={styles.text}>{goal}</Text>
        </TouchableOpacity>
      ))}

      <Button
        title="Continue"
        onPress={() => navigation.navigate('Challenge')} // temporary
      />
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
    backgroundColor: '#c7f9cc',
    borderColor: '#38b000',
  },
  text: {
    fontSize: 16,
  },
});