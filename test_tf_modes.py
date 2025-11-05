"""
test_tf_modes.py
Test TensorFlow Laplacian & Biharmonic operator implementations
"""
import tensorflow as tf
from models_tf import MLP
from operators_tf import laplacian_operator, biharmonic_operator

def test_laplacian_tf():
    print("=" * 60)
    print("Test 1: TensorFlow Laplacian Operator (MLP)")
    print("=" * 60)
    input_dim = 3
    model = MLP(input_dim=input_dim, hidden_dims=[32, 32], output_dim=1, activation='tanh')
    x = tf.random.normal((1, input_dim), dtype=tf.float32)
    val = laplacian_operator(model, x)
    print(f"✓ Laplacian (reverse_reverse): {val.numpy():.8f}")

def test_biharmonic_tf():
    print("\n" + "=" * 60)
    print("Test 2: TensorFlow Biharmonic Operator (MLP)")
    print("=" * 60)
    input_dim = 3
    model = MLP(input_dim=input_dim, hidden_dims=[32, 32], output_dim=1, activation='tanh')
    x = tf.random.normal((1, input_dim), dtype=tf.float32)
    val = biharmonic_operator(model, x)
    print(f"✓ Biharmonic (reverse_reverse): {val.numpy():.8f}")

if __name__ == "__main__":
    test_laplacian_tf()
    test_biharmonic_tf()
    print("\nAll TensorFlow operator tests complete!")
