import os

def keep_middle_line_in_folder(folder_path):
    # List all text files in the specified folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):  # Process only .txt files
            file_path = os.path.join(folder_path, filename)
            
            with open(file_path, 'r') as f:
                lines = f.readlines()

            # Calculate the middle index
            num_lines = len(lines)
            if num_lines == 0:
                print(f"The file {filename} is empty. Skipping.")
                continue

            middle_index = (num_lines - 1) // 2  # First middle line if even

            # Keep only the middle line
            middle_line = lines[middle_index]

            # Overwrite the original file with the middle line
            with open(file_path, 'w') as f:
                f.write(middle_line)

            print(f"Processed {filename}: kept line {middle_index + 1}")

# Example usage
folder_path = './images'  # Replace with your folder path
keep_middle_line_in_folder(folder_path)

