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

- Make sure you have chromedriver installed. Following command will automatically create a .txt file in txt/ which will set the path for the chromedriver for the repo:
```
make driver_path
```

- You need to create 'API_key.ini' with API_keys for Youtube and Google scholar. Use 'API_key_template.ini' file as a template. Make sure 'API_key.ini' is listed in '.gitignore', so you will NEVER share your API_key to the public.

## Additional Setup for Windows Users
- You have to use WSL2 in order to run linux terminal commands. You can find detailed tutorial for installing WSL2 [here](https://learn.microsoft.com/en-us/windows/wsl/install). In short, you simply have to run following command on Windows PowerShell:
```
wsl --install
```

- Then, you have to install Chrome on WSL2. Follow the instruction under "Install Google Chrome for Linux" in this [link](https://learn.microsoft.com/en-us/windows/wsl/tutorials/gui-apps).

- You can now follow the instructionson 'Setup' to install Miniconda and import/activate the environment for the repo.


## Testing Demo with ArXiv

- Use following command to scrape through arXiv with manually using keywords from [keywords.txt](txt/keywords.txt):
```
make search_arxiv
```
- In order to predefine your own initial keywords to scrape, change the context of [keywords.txt](txt/keywords.txt). 
- If you want to scrape through different search engines, pick one of the following commands instead of 'search_arxiv:
```
make search_scholar
make search_MLM
make search_medium
make search_youtube
```

- Use following command to run the initial clustering using the [Arxiv_Resources.csv file](data/Arxiv_Resources.csv) scraped from the previous step:
```
make test
```

- Use following command to iterate the scraping and clustering on every clusters created by the initial cluster from the previous step:
```
make test_iteration
```

- In order to change any of the input values, such as input/output file directory and any of the hyperparameters, check the [Makefile](Makefile) and open the corresponding .py file to change the input values. Input values are state at the end of the .py files below 'If __name__ == "__main__"'

- If the process is taking too long, you may skip some of the processings. For example, if don't apply lemmatization, which can be found on line 409 of [test.py](src/test.py), the computational time will be shorter.

- Whenever you need a fresh-start, use following command to remove generated data, txt files and figures:
```
make clean
```

## Analysis with Colab
- If you are more used to Colab or Jupyter environment, you may check the [.ipynb file](src/FinalProject(Finalized).ipynb). This file is meant to be ran at Colab, and the current environment wouldn't work due to TensorFlow used in the file. Please use Colab environment, or use Jupytern with the same package dependencies as Colab.

- Check [docs](docs/) for EDA and presentations.