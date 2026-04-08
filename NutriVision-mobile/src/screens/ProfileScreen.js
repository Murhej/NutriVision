import React, { useState } from 'react';
import { View, Text, StyleSheet, Modal, TextInput, TouchableOpacity } from 'react-native';
import { Button } from '../components';

export default function ProfileScreen() {
  const [showModal, setShowModal] = useState(false);

  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirm, setConfirm] = useState('');

  const handleSignup = () => {
    if (!username || !email || !password || !confirm) {
      alert('Fill all fields');
      return;
    }

    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
      alert('Enter a valid email');
      return;
    }

    const passwordRegex = /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&]).{8,}$/;
    if (!passwordRegex.test(password)) {
      alert('Password must include:\n- 8+ characters\n- Uppercase\n- Lowercase\n- Number\n- Special character');
      return;
    }

    if (password !== confirm) {
      alert('Passwords do not match');
      return;
    }

    alert('Account Created!');
    setShowModal(false);
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Your Profile</Text>

      <Text>Name: Albin</Text>
      <Text>Age: 23</Text>
      <Text>Gender: Male</Text>

      <Button
        title="Start Your Journey"
        onPress={() => setShowModal(true)}
      />

      <Modal visible={showModal} animationType="slide">
        <View style={styles.modal}>
          <Text style={styles.modalTitle}>Sign Up</Text>

          <TextInput
            placeholder="Username"
            style={styles.input}
            value={username}
            onChangeText={setUsername}
          />

          <TextInput
            placeholder="Email"
            style={styles.input}
            value={email}
            onChangeText={setEmail}
          />

          <TextInput
            placeholder="Password"
            style={styles.input}
            secureTextEntry
            value={password}
            onChangeText={setPassword}
          />

          <TextInput
            placeholder="Confirm Password"
            style={styles.input}
            secureTextEntry
            value={confirm}
            onChangeText={setConfirm}
          />

          <Button title="Sign Up" onPress={handleSignup} />

          <TouchableOpacity onPress={() => alert('You must sign up to continue')}>
            <Text style={styles.skip}>Maybe Later</Text>
          </TouchableOpacity>
        </View>
      </Modal>
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
  },
  modal: {
    flex: 1,
    justifyContent: 'center',
    padding: 20,
  },
  modalTitle: {
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
  skip: {
    marginTop: 15,
    textAlign: 'center',
    color: 'gray',
  },
});