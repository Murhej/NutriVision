import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { Button } from '../components';

export default function JourneyStartScreen({ navigation }) {
  return (
    <View style={styles.container}>
      <Text style={styles.title}>Login Details</Text>

      <View style={styles.card}>
        <Text style={styles.label}>Your account is ready.</Text>
        <Text style={styles.info}>You have completed the onboarding steps.</Text>
        <Text style={styles.info}>Press the button below to continue.</Text>
      </View>

      <Button
        title="Start Your Journey"
        onPress={() => navigation.replace('MainTabs')}
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
    fontSize: 24,
    marginBottom: 20,
    textAlign: 'center',
    fontWeight: 'bold',
  },
  card: {
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 12,
    padding: 20,
    marginBottom: 20,
    backgroundColor: '#fff',
  },
  label: {
    fontSize: 18,
    fontWeight: '600',
    marginBottom: 10,
  },
  info: {
    fontSize: 16,
    marginBottom: 6,
  },
});