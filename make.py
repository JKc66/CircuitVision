import os

def combine_files_markdown_style(input_files, output_file, base_dir="."):
    base_dir = os.path.abspath(base_dir)

    with open(output_file, 'w', encoding='utf-8') as out:
        for file_path in input_files:
            try:
                abs_path = os.path.abspath(file_path)
                rel_path = os.path.relpath(abs_path, base_dir)

                with open(abs_path, 'r', encoding='utf-8') as f:
                    out.write(f"```{rel_path}\n")
                    out.write(f.read())
                    out.write("\n```\n\n")
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

if __name__ == "__main__":
    # Base directory for relative paths
    base_directory = "."

# List of input files (can be absolute or relative paths)
files_to_combine = [
    "src/circuit_analyzer.py",
    "src/utills.py",
    "src/sam2_infer.py",
    "app.py"
]
output_filename = "combined_output.md"

combine_files_markdown_style(files_to_combine, output_filename, base_directory)
print(f"Combined content saved to {output_filename}")
