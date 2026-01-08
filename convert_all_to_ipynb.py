import os
import nbformat as nbf

# Folder containing your .py files
folder_path = r"C:\Users\gopic\Pinecone"  # <-- your current folder

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".py"):
        py_path = os.path.join(folder_path, filename)
        notebook_name = filename.replace(".py", ".ipynb")
        ipynb_path = os.path.join(folder_path, notebook_name)

        # Read the Python file
        with open(py_path, "r", encoding="utf-8") as f:
            code = f.read()

        # Create a new notebook and add the code
        nb = nbf.v4.new_notebook()
        nb.cells.append(nbf.v4.new_code_cell(code))

        # Save as .ipynb
        with open(ipynb_path, "w", encoding="utf-8") as f:
            nbf.write(nb, f)

        print(f"Converted: {filename} -> {notebook_name}")

print("All .py files have been converted to .ipynb notebooks.")
