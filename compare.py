import numpy as np
import os
import sys
try:
    from scipy.io import loadmat
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. MAT file support disabled.")

def load_matrix(filename):
    """Load matrix from .npy or .mat file"""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} not found")
    
    if filename.endswith('.npy'):
        return np.load(filename)
    elif filename.endswith('.mat'):
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy is required to load MAT files. Install with: pip install scipy")
        
        mat_data = loadmat(filename)
        # Remove MATLAB metadata keys and get the actual data
        data_keys = [k for k in mat_data.keys() if not k.startswith('__')]
        
        if len(data_keys) == 1:
            return mat_data[data_keys[0]]
        elif len(data_keys) == 0:
            raise ValueError(f"No data found in MAT file {filename}")
        else:
            print(f"Multiple variables found in {filename}: {data_keys}")
            print("Using the first variable.")
            return mat_data[data_keys[0]]
    else:
        raise ValueError(f"Unsupported file format. Use .npy or .mat files only.")

def compare_matrices(file1, file2, tolerance=1e-9):
    """Compare two matrix files and report differences"""
    print(f"Comparing {file1} and {file2}")
    print("-" * 50)
    
    try:
        # Load both matrices
        matrix1 = load_matrix(file1)
        matrix2 = load_matrix(file2)
        
        print(f"Matrix 1 shape: {matrix1.shape}")
        print(f"Matrix 2 shape: {matrix2.shape}")
        print()
        
        # Check if shapes match
        if matrix1.shape != matrix2.shape:
            print("❌ MATRICES DO NOT MATCH")
            print(f"Shape mismatch: {matrix1.shape} vs {matrix2.shape}")
            return False
        
        # Check if matrices are identical (within tolerance for floating point)
        if np.allclose(matrix1, matrix2, atol=tolerance):
            print("✅ MATRICES MATCH")
            print("All elements are identical (within tolerance)")
            return True
        else:
            print("❌ MATRICES DO NOT MATCH")
            
            # Find differences
            diff_mask = ~np.isclose(matrix1, matrix2, atol=tolerance)
            diff_indices = np.where(diff_mask)
            
            print(f"Number of differing elements: {np.sum(diff_mask)}")
            print("\nDifferences found at the following indices:")
            print("(row, col): matrix1_value -> matrix2_value")
            print("-" * 40)
            
            # Show up to 20 differences to avoid overwhelming output
            max_diffs_to_show = 20
            num_diffs = len(diff_indices[0])
            
            for i in range(min(num_diffs, max_diffs_to_show)):
                row, col = diff_indices[0][i], diff_indices[1][i]
                val1, val2 = matrix1[row, col], matrix2[row, col]
                print(f"({row}, {col}): {val1} -> {val2}")
            
            if num_diffs > max_diffs_to_show:
                print(f"... and {num_diffs - max_diffs_to_show} more differences")
            
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return False

def main():
    """Main function to handle command line arguments"""
    if len(sys.argv) != 3:
        print("Usage: python matrix_comparator.py <file1> <file2>")
        print("\nSupported formats: .npy, .mat")
        print("\nExamples:")
        print("python matrix_comparator.py matrix1.npy matrix2.npy")
        print("python matrix_comparator.py data1.mat data2.npy")
        return
    
    file1, file2 = sys.argv[1], sys.argv[2]
    
    # Set tolerance for floating point comparison
    tolerance = 1e-9  # You can adjust this value as needed
    
    result = compare_matrices(file1, file2, tolerance)
    
    # Exit with appropriate code
    sys.exit(0 if result else 1)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("No command line arguments provided.")
        print("Usage: python matrix_comparator.py <file1> <file2>")
        print("Supported formats: .npy, .mat")
    else:
        main()