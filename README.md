# DS5230-Group1-FinalProject

**Objective**
Develop a system to group similar resources (e.g., articles, videos, or blog posts) into clusters
based on their content. The goal is to discover meaningful groupings that can guide students in
selecting resources by topic or difficulty level.

Project sprint descriptions in docs/project_plan.md

## Setup
- The repo is based on Linux terminal command-line with use of 'make.' Please refer to [this page](https://github.com/ds5110/git-intro/blob/main/setup.md) for further discussion of benefits in reproducibility by doing so. Highly suggest installing miniconda using your Linux terminal as it is described [here](https://www.anaconda.com/docs/getting-started/miniconda/install).

- Following command will import the conda environment:
```
conda env create -f environment.yml
```

- Then, activate the conda environment with the following command:
```
conda activate ds5230
```

## Additional Setup for Windows Users
- You have to use WSL2 in order to run linux terminal commands. You can find detailed tutorial for installing WSL2 [here](https://learn.microsoft.com/en-us/windows/wsl/install). In short, you simply have to run following command on Windows PowerShell:
```
wsl --install
```

- Then, you have to install Chrome on WSL2. Follow the instruction under "Install Google Chrome for Linux" in this [link](https://learn.microsoft.com/en-us/windows/wsl/tutorials/gui-apps).

- You can now follow the instructionson 'Setup' to install Miniconda and import/activate the environment for the repo.


## Testing with Google Scholar.

- Use following command to avoid manually solving CAPTHA:

```
make cookie
```

- Use following command to search through google scholar's abstract:

```
make search_scholar
```
- This is going to display on command line unique words that were shown from the searched papers that their abstracts were retrieved.
