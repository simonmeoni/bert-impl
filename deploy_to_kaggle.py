# openfile
with open('src/bert_implementation.ipynb', 'r') as f:
    file_source = f.read()
    replace_dependencies = file_source.replace("Python 3.7.9 64-bit ('bert': conda)", 'python3')
    remove_neptune_import = replace_dependencies.replace('"import neptune\\n",', '')
    replace_python_kernel = remove_neptune_import.replace(
        '"NEPTUNE_API_TOKEN = os.environ.get(\\"NEPTUNE_API_TOKEN\\")\\n",', """
    "!pip install neptune-client\\n",
    "from kaggle_secrets import UserSecretsClient\\n",
    "user_secrets = UserSecretsClient()\\n",
    "NEPTUNE_API_TOKEN = user_secrets.get_secret(\\\"NEPTUNE_API_TOKEN\\\")\\n",
    "import neptune\\n",
        """)
with open('src/bert_implementation_to_kaggle.ipynb', 'w') as f:
    f.write(replace_python_kernel)
