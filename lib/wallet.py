import os
import io

def read_file (filepath):
    content = ""
    with io.open(filepath, "r") as f: 
        content = f.read()
    return content

def write_file (filepath, line):
    with io.open(filepath, "r") as f: 
        f.write(str(line)+"\n")

def write_to_file (filepath, line):
    with io.open(filepath, "a") as f: 
        f.write(str(line)+"\n")

def get_secret_folder():
    folders = []
    folders.append(os.path.abspath(os.path.join(os.sep, "secret")))
    folders.append(os.path.join("secret"))
    for folder in folders:
        check_file = os.path.isfile(os.path.join(folder, "wallet.pub.txt"))
        check_file = check_file & os.path.isfile(os.path.join(folder, "wallet.pk.txt"))
        if check_file: break
    else:
        folder = None 
    
    return folder

def get_public_key():
    secret_folder = get_secret_folder()
    filepath = os.path.join(secret_folder, "wallet.pub.txt")
    return read_file(filepath)

def get_private_key():    
    secret_folder = get_secret_folder()
    filepath = os.path.join(secret_folder, "wallet.pk.txt")
    return read_file(filepath)

