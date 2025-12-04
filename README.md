# Agentic Playground

IMAGE

### Prerequisites - Install Required Software

* Open Command Prompt or PowerShell

#### Checking python version (should be greater than or equal to 3.12 and strictly less than 3.13 for CrewAI installation later)
* python --version 

#### Checking Git version 
* git --version

Refer to the following for installation if needed:
* Python: Download from python.org (greater than or equal to 3.12 and less than 3.13)
* Git: Download from git-scm.com
* VSCode: Download from code.visualstudio.com

### Install UV (Python Package Manager)

By default, uv is installed to a path like ~/.cargo/bin on Unix-like systems, which may require the user to close and reopen their terminal (or run a shell-specific command like source ~/.bashrc) to have the uv command available globally in that session.

**Note:** After running the installation script, you may need to close and reopen your Command Prompt/PowerShell/Terminal or run a command like source ~/.bashrc (if applicable) for the uv command to be recognized globally.

#### On Windows (PowerShell) 
* powershell -c "irm https://astral.sh/uv/install.ps1 | iex" 

#### On Mac/Linux
* curl -LsSf https://astral.sh/uv/install.sh | sh

#### Verify installation
* uv --version

### Clone the Repository
* Open VS Code terminal and run cd ~/Documents # or wherever you prefer 

#### Clone the repository 
* git clone https://github.com/gitmystuff/agentic-playground.git 

#### Navigate into the project 
* cd agentic-playground

### Setup Environment 

* Run command uv self update 
* **Deactivate Anconda -> conda deactivate** Important if you use Anaconda 

#### 1\. Open the Project Folder in VS Code

Open the **VS Code** application.

  * Go to **File** in the top menu bar.
  * Select **Open Folder...** (or **Open...** on macOS).
  * Navigate to and select the folder named **agentic-playground**.

#### 2\. Open the Integrated Terminal

Once the folder is open in VS Code, open the integrated terminal:

  * Go to **Terminal** in the top menu bar.
  * Select **New Terminal**.
      * *Alternatively, you can use the keyboard shortcut:* $\text{Ctrl} + \text{\`}$ (Windows/Linux) or $\text{Cmd} + \text{\`}$ (macOS).

#### 3\. Confirm Your Directory (Working Directory)

The terminal's prompt should show you are currently inside the **agentic-playground** folder.

  * **Check:** Look at the path displayed in the terminal. You need to verify that the file named **`pyproject.toml`** is directly visible in this current directory.
  * **Command (Optional Check):** You can run the following command to list the contents of the current directory and confirm the file is there:
      * On Windows/Linux/macOS: `ls`
      * On Windows (if `ls` isn't available): `dir`

#### 4\. Run `uv sync`

With the terminal open and confirmed to be in the correct folder (the one containing `pyproject.toml`), execute the following command:

```bash
uv sync
```

This command uses the **`uv`** package manager to synchronize your project's dependencies based on the configuration in your `pyproject.toml` file.

#### 5\. Uv will now install everything (may take a while)

### Install CrewAI

* Run uv tool install crewai (Note: CrewAI requires Python versions greater than or equal to 3.12 and strictly less than 3.13)
* Run uv tool list to verify CrewAI
* Run cls (or clear)
