# TERMINAL CHEAT SHEET

## Basic Commands
- `pwd` - Print working directory

- `ls` - List directory contents
- `cd <directory>` - Change directory
- `cd ~` - Change to home directory
- `cd ..` - Change to parent directory
- `mkdir <directory>` - Create a new directory
- `rm <file>` - Remove a file
- `rmdir <directory>` - Remove an empty directory
- `rm -r <directory>` - Remove a directory and its contents recursively
- `cp <source> <destination>` - Copy files or directories
- `mv <source> <destination>` - Move or rename files or directories
- `mv old-directory new-directory` - rename
- `touch <file>` - Create an empty file or update the timestamp of an existing file
- `cat <file>` - Concatenate and display file content
- `file <file>` - Determine the file type

## kill process
`ps aux | grep extract_batches.py`
- `ps aux`: Lists all running processes on the system, showing details like user, PID, CPU/memory usage, and command.
- `|`: Pipes the output of ps aux to the next command.
- `grep extract_batches.py`: Searches (filters) the output for lines containing extract_batches.py.
**Result:**
You see all running processes that include extract_batches.py in their command line—i.e., if your script is running, it will show up here.
```bash
root       11764  2.2 22.9 3820180 3637616 pts/1 Sl+  15:46   1:51 python extract_batches.py
root       11885  0.0  0.0   6420  2192 pts/2    S+   17:10   0:00 grep --color=auto extract_batches.py
```
**HOW TO KILL**
kill: `kill 11764`
force kill: `kill -9 11764`

## File Permissions
- `chmod <permissions> <file>` - Change file permissions
  - Example: `chmod 755 script.sh` (rwxr-xr-x)

## Searching Files
- **find folder in directory**
  ```bash
  find /path/to/search -type d -name "sat-3l"
  ```
- **find by name**
  ```bash
  find /path/to/search -name "filename"
  ```
- **find by name in current directory**
  ```bash
  find . -name "filename"
  ```
- **search file contents**
  ```bash
  grep -r "search_term" /path/to/search
  ```

## Show File Structure
```bash
tree /path/to/directory
```
Customize how many levels
```bash
tree -L <level> /path/to/directory
```


## Memory/Disk Usage
- `df -h` - Display disk space usage in human-readable format
- `free -h` - Display memory usage in human-readable format

## SSH 
**Connect:**
`ssh admin@ip`
### Port Forwarding
From your local machine, run:
```bash
ssh -L 8080:localhost:8080 youruser@your-vm-ip
ssh -L 8080:localhost:8080 admin@ip
```

Leave this SSH session open.
Now, visit http://localhost:8080/docs on your local browser.
You’ll see the Swagger UI as if you were on the VM.





## Copying Files
- **copy from windows to linux server**

  write this in Windows PowerShell or Command Prompt:

  ```bash
  scp -r C:\Users\username\directory admin@ip:/home/admin/
  ```

- **copy from wsl to linux server**
  ```bash
  scp file.tar admin@ip:/home/admin/directory_destination
  #directory
  scp -r ~/hfcache admin@ip:/home/admin/directory_destination
  ```
- **copy from linux server to windows**
  ```bash
  scp -r admin@ip:/home/admin/project C:\Users\username\Downloads
  ```

- **copy from WSL to Windows**
Write in windows powershell:
  ```bash
  cp -r ~/project_directory /mnt/c/Users/username/
  ```
- **copy from windows to WSL**

  write in wsl terminal:
  ```bash
  cp -r /mnt/c/Users/username/Downloads/file.tar.gz ~/
  ```

## VS Code issues in WSL
If you encounter issues with VS Code not recognizing your WSL environment, try the following:
  1. Ensure you have the "Remote - WSL" extension installed.
  2. Open a WSL terminal and run `code .` to launch VS Code in the current directory.
  3. If you still face issues, try reloading the WSL integration by running the command palette (Ctrl+Shift+P) and selecting "Remote-WSL: Reopen Folder in WSL".
  4. Deleting ~/.vscode-server forces VS Code to reinstall its server components in WSL, which often fixes stubborn issues with extensions and connectivity.
  ```bash 
  rm -rf ~/.vscode-server
  ```



# Docker Commands

the image is loaded from tar and when compose up is called the image is turned into a container

## Build Docker Images
```bash
docker build -t <image_name> .
``` 
To specify the file, --file and is used to specify the path to the Dockerfile you want to use for building the image. to specify the nam/tag -t is used
```bash
docker build -f deployment/worker.Dockerfile -t osddy-worker .
```

## Save Docker Images

```bash
docker save -o <output_file.tar> <image_name>
docker save -o osddy-worker.tar osddy-worker
```

## Remove previous docker containers and images
**remove container**
```bash
docker rm <container_name>
```
**remove image**
```bash
docker rmi <image_name>
```
**remove all exited containers**
```bash
sudo docker container prune
```

**remove dangling images**
```bash
docker rmi $(docker images -f "dangling=true" -q)
```
what are dangling images? 
Dangling images are layers that have no relationship to any tagged images. They are typically created during the image build process and can take up unnecessary space on your system.

## Load Docker Images

```bash
docker load -i <input_file.tar>
```

### load and run
```bash
sudo docker load -i myworker.tar
sudo docker load -i my-api-image.tar
sudo docker load -i redis.tar
sudo docker compose up -d
```

## Run Docker Containers

**docker compose up** - This command is used to start all the services defined in a `docker-compose.yml` file.

```bash
docker run --name <container_name> -d <image_name>
```

## monitoring docker containers
```bash
docker ps -a
docker logs <container_name>
docker stats
```

# Python virtual environment

## Anaconda
```bash
conda create -n myenv python=3.8
conda activate myenv
```
list envs
```bash
conda env list
```
## venv
```bash
python -m venv myenv
source myenv/bin/activate  # On Windows use: myenv\Scripts\activate 
pip install pandas numpy xgboost scikit-learn
```

## Poetry
**create new environment**
```bash
poetry new my-nlp-project
cd my-nlp-project
# add dependencies
poetry add flair torch fasttext
# run python files
poetry run python model.py
```

**shell with environment activated**
```bash
poetry env info --path
#result
source <env_path>/bin/activate
```
**run command in poetry env**
```bash
# Run a command inside the poetry environment
poetry run <command>
# Example:
poetry run python script.py
```

### TIP FOR MACHINE LEARNING PROJECTS
**Best practice:**
When working with machine learning projects, manually set an upper bound for Python to match PyTorch’s/triton’s supported versions:

`TOML file:`
```bash
requires-python = ">=3.11,<3.14"
```
**make libraries use cpu only**
```python
flair.device = torch.device('cpu') ---> Αυτούσιο έτσι
stanza.Pipeline(lang='en', device='cpu') ----> καπου θα δεις να υπαρχει αυτο και θα πρεπει να του βαλεις το device
```
