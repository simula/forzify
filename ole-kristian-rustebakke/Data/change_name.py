import os

folder_path = "/itf-fi-ml/home/olekrus/master/master/Data/val/labels"  # Replace with your folder path

# List all files in the folder
for filename in os.listdir(folder_path):
    # Check if the file starts with 'txt'
    if filename.startswith("txt"):
        # Create the new filename by replacing the first three letters
        new_filename = "img" + filename[3:]
        
        # Construct full paths
        old_file_path = os.path.join(folder_path, filename)
        new_file_path = os.path.join(folder_path, new_filename)
        
        # Rename the file
        os.rename(old_file_path, new_file_path)
        print(f"Renamed {filename} to {new_filename}")
