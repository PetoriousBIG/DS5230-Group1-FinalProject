# DS5230-Group1-FinalProject

**Objective**
Develop a system to group similar resources (e.g., articles, videos, or blog posts) into clusters
based on their content. The goal is to discover meaningful groupings that can guide students in
selecting resources by topic or difficulty level.

Project sprint descriptions in docs/project_plan.md

## Setup
- The repo is based on Linux terminal command-line with use of 'make.' Please refer to [this page](https://github.com/ds5110/git-intro/blob/main/setup.md) for further discussion of benefits in reproducibility by doing so. 

- Following command will import the conda environment:
```
conda env create -f environment.yml
```

- Then, activate the conda environment with the following command:
```
conda activate ds5230
```

## Testing with Google Scholar.

- Use following command to avoid manually solving CAPTHA:
````
make cookie
```

- Use following command to search through google scholar's abstract:
```
make search_scholar
```
- This is going to display on command line unique words that are being presented on each abstract.
