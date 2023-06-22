# PythonProjectTemplate

Repository containing scaffolding for a Python 3-based data science project. The template is actually an adapted version of the original poject repository from [python-data-science-project](https://github.com/KAUST-Academy/python-data-science-project). A good resource to start with coding conventions is to read the PEP8 from the developers at Python [PEP8 style guide](https://peps.python.org/pep-0008/). One of the great practices can be discovered through this list of aphorisms [PEP8 Zen of Python](https://peps.python.org/pep-0020/)

## First Things first

Welcome to IWES Wind! For registering with or accessing [Gitlab](https://gitlab.cc-asp.fraunhofer.de/), you need to do the following:

1. Setup a secondary username and password. You would need a secondary username and password which can set in the [self-service directory](https://www.directory.fraunhofer.de/). Login name (Max23456), password (Fraunhofer password generally). Note that yor windows login/ username, e.g. `MusMax` won't work on Gitlab.
2. install [Git](https://git-scm.com/downloads) on your PC
3. [Generate a ssh key](https://docs.gitlab.com/ee/user/ssh.html#generate-an-ssh-key-pair).The SSH stands for Secure Socket Shell used for authenticaticating GitLab server without using username and password each time. (for windows quick sample here: https://www.maketecheasier.com/generate-ssh-public-private-keys-windows/)
4. [Add a SSH ley to your Gitlab account](https://docs.gitlab.com/ee/user/ssh.html#add-an-ssh-key-to-your-gitlab-account)

## Creating a new project from this template

Simply follow the [instructions](https://help.github.com/en/articles/creating-a-repository-from-a-template) to create a new project repository from this template. 

OR

Open cmd on your personal computer and type the following command to clone the drive in your local drive

```bash
cd path\to\destination\folder 
git clone https://gitlab.cc-asp.fraunhofer.de/iwes_wind/PythonProjectTemplate
```

Once, the Project template is created, you may wish to delete previous commit history and remove files that you feel are unnecessary and save with a new name to your repository. Follow the steps below by opening a shell within the project folder:

remove the history from git folder (type `bash` on cmd and `enter`, `exit` brings you back to cmd shell). if bash is not installed on your system, write the commented option
```bash
rm -rf .git
# del .git 
```

Delete the unnecessary objects in the project folder. Re-initiate the repository
```bash
git init
git add .
git commit -m "Initial commit"
```

Push the project folder into your repository on gitlab `iwes_wind` or your personal account, e.g. `MusMax`:
```bash
git remote add origin git@gitlab.cc-asp.fraunhofer.de:iwes_wind/<RepositoryName>.git
git push -u --force origin master
```

Alternatively, if you are a visual person. Install [Sourcetree](https://www.sourcetreeapp.com/) and follow the tutorials to get started with [Version control](https://confluence.atlassian.com/get-started-with-sourcetree/install-and-set-up-sourcetree-847359043.html). The same steps will be followed on sourcetree

## Project organization

Project organization is based on ideas from [_Good Enough Practices for Scientific Computing_](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005510).

1. Put each project in its own directory, which is named after the project.
2. Put external scripts or compiled programs in the `bin` directory.
3. Put raw data and metadata in a `data` directory.
4. Put text documents associated with the project in the `doc` directory.
5. Put all Docker related files in the `docker` directory.
6. Install the Conda environment into an `env` directory. 
7. Put all notebooks in the `notebooks` directory.
8. Put files generated during cleanup and analysis in a `results` directory.
9. Put project source code in the `src` directory.
10. Put test code in the `test` directory.
11. Name all files to reflect their content or function.

## Using Conda

1. Install [Anaconda](https://docs.anaconda.com/anaconda/install/windows/) (heavier distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) on your PC.
2. Create environments where your python interpretor will run and use the standard/ imported libraries. Follow [Manage Environments](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). As an example, I have three env
    - base (backup environment if something fails)
    - py39 (python environment with py3+ versions)
    - py27 (python environment with older python scripts)
3. Building an environment based on .yml file or .txt file

### Creating the Conda environment

After adding any necessary dependencies for your project to the Conda `environment.yml` file 
(or the `requirements.txt` file or the explicit `environment.txt`), you can create the environment in a directory, usually `c:\Users\[name]\Miniconda3\envs\` by running the following command.

```bash
conda create --name py39 --file environment.txt # or
conda env create --file environment.yml --force
```

Once the new environment has been created you can activate the environment with the following 
command.

```bash
conda activate py39
```

For your convenience these commands have been combined in a shell script `./bin/create-conda-env.sh`. 
Running the shell script will create the Conda environment and activate the Conda environment

```bash
./bin/create-conda-env.sh
```

If you want to share an environment, the following steps as given on the following link shall help you. I just give a core command to create the environment.yml file to share
[Share an environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#sharing-an-environment).

```bash
conda env export > environment.yml
```

The environment file then looks like. The first run command is what you add into shell/bash/conda prompt. The name of the environment is taken from the environment.yml file as default.

```
# run: conda env create --file environment.yml
name: test-env
dependencies:
- python>=3.5
- anaconda
- pip
- numpy=1.13.3  # pin version for conda
- pip:
  # works for regular pip packages
  - docx
  - gooey
  - matplotlib==2.0.0  # pin version for pip
  # and for wheels
  - http://www.lfd.uci.edu/~gohlke/pythonlibs/bofhrmxk/opencv_python-3.1.0-cp35-none-win_amd64.whl
```


### Listing the full contents of the Conda environment

The list of explicit dependencies for the project are listed in the `environment.yml` file. To see 
the full lost of packages installed into the environment run the following command.

```bash
conda update conda
conda env list # list environments
conda list  # list packages in active env
conda list --revisions # history of changes
conda list --revisions 2 # restore to a previous revision
```

Add conda-forge as the highest priority channel.

```bash
conda config --add channels conda-forge
```
Activate strict channel priority (strict will be activated by default in conda 5.0).

```bash
conda config --set channel_priority strict
```

`Note: In addition to the channel priority, we recommend always installing your packages inside a new environment instead of the base (formerly known as root) environment, and we also recommend the use of miniconda instead of the Anaconda Distribution. Using environments make it easier to debug problems with packages and ensure the stability of your base environment. Avoiding the Anaconda Distribution reduces chances of unsolvable/conflicting installations, it is also a smaller download.`

`Note: Please be aware that the order of your conda package channels is important, especially when you combine conda-forge with other channels, e.g. bioconda.`

`Note: There are more ways to manage the libraries/modules for python: Poetry or pip. Poetry is very new and we don't have much experience with it. Poetry helps in resolving the problems easily and is detailed below as an additional section. Pip is as good as conda, except that conda-forge maintains the dependencies, somewhat like poetry. Maybe we add a section on Pip soon!`

## Using Poetry
In order to use Poetry, please follow the updated information on the page [Poetry](https://python-poetry.org/docs/). For Windows, I repeat some of the processes here:

1) Install Poetry by pasing the following on windows powershell

```shell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
```
`If you have installed Python through the Microsoft Store, replace py with python in the command above.`

2) Add poetry to your path. On Windows this is 
`%APPDATA%\Python\Scripts`

3) Check Poetry version. if this works, you have installed poetry properly
```shell
poetry --version
poetry self update
```
`You may need to restart shell in order for the changes to take effect`

4 Configuring poetry
There are certain configurations, that help  maintaining poetry in the way you might like to. An overview of the config options can bee seen using, and the general usage of config follows later

```shell
poetry config --list
poetry config [options] [setting-key] [setting-value1]
```
`[options] has has one valuable option --local, which can be used to set/get settings specific to a project in the poetry.toml file`. 

The results of poetry config --list` are given as follows:

```shell
cache-dir = "C:\\Users\\<user_name>\\AppData\\Local\\pypoetry\\Cache"
experimental.new-installer = true     
experimental.system-git-client = false
installer.max-workers = null
installer.no-binary = null
installer.parallel = true
virtualenvs.create = true
virtualenvs.in-project = null
virtualenvs.options.always-copy = false
virtualenvs.options.no-pip = false
virtualenvs.options.no-setuptools = false
virtualenvs.options.system-site-packages = false
virtualenvs.path = "{cache-dir}\\virtualenvs"  # C:\Users\<user_name>\AppData\Local\pypoetry\Cache\virtualenvs
virtualenvs.prefer-active-python = false
virtualenvs.prompt = "{project_name}-py{python_version}"
```

The default location of poetry env is given by virtualenvs.path (see above). In order to create local virtual env stored in the same root folder of PythonProjectTemplate, type in the following code

```shell
poetry config virtualenvs.in-project true # comment this if you want your virtuallenv to be in virtualenv.path
poetry install
```

Instead of just using the poetry.toml and poetry.lock file within this project template folder, you can also initialize the poetry file using the following code from scratch in a new folder/project. The following commands create the poetry.toml and poetry.lock file. You can then add packages similarly to conda/pip install commands as follows:

```shell
cd pre-existing-project
poetry init
poetry add package-name
```

Usually, the poetry environment is automatically activated depending on the poetry.toml file. You can activate your environment and deactivate your environnment using the following commands

```shell
poetry shell
deactivate
```
To deactivate the virtual environment and exit this new shell type exit. To deactivate the virtual environment without leaving the shell use deactivate.

For detailed explanations and troubleshooting refer to the poetry documentation. Please come back to me if there is incorrect implementation in any way or an improvement.

## Using Docker [expert level, avoid as much as possible]
In order to build Docker images for your project and run containers you will need to install 
[Docker](https://docs.docker.com/install/) and [Docker Compose](https://docs.docker.com/compose/install/).

Detailed instructions for using Docker to build and image and launch containers can be found in 
the `docker/README.md`.

## Building Tests in Python

A good practice is to start with tests before creating a script. It could be a basic test of creating a fictitious variable/data and testing if the expected value is achieved from a test or an advanced unit test based script. A good source of python tutorials on testing can be found on [realpython](https://realpython.com/python-testing/)

## Version control with Git and IWES gitlab

Even if you haven't coded much, it is important to learn to learn version controlling using Git. I am not only using Git for code, rather also for latex documents or any text documents. Beware, do not perform version controlling on data (my first mistake on Git). The CFD colleagues at IWES organized a workshop on Git and gave a hands-on experience on Git. You can follow the presentation that are very well illustrated learn [how to git](https://gitlab.cc-asp.fraunhofer.de/iwes-cfsd/teaching/how-to-git.git). The instructions to setup the IWES Gitlab are also included within the presentation.

Further reading and troubleshooting refer to two main sources:
https://sethrobertson.github.io/GitBestPractices/
https://eagain.net/articles/git-for-computer-scientists/
https://jwiegley.github.io/git-from-the-bottom-up/

## Using VS Code and functionalities
VS Code is very versatile when used for data analysis and python coding at the same time.

PROS: 
-   One can use VS Code as all in one solution for managing environment, writing code and using other applications (using viewers and extensions)
-   The setting files can be shared, meaning that the workflow is transferrable

CONS:
-   is a bit sluggish compared to spyder and other python IDEs
-   One should understand how configurations are performed on IDE, when one is used to Spyder+Anaconda or PyCharm implementations

The user settings.json and keybindings.json are provided in the ./bin folder in this repo. Please add them to the VS Code settings to configure your VS Code environment. You can then adapt it to suit your needs.

To export the user extensions, please use the following commands on shell cmd line

```shell
code --list-extensions > ./bin/extlist
```
In order to install all the extensions listed, one can use the following code on shell cmd line

```shell
for /f %f in (./bin/extlist) do code --install-extension %f
```

or copy-paste individual lines from the [extension_list](bin\extensions_list) file in the bin folder.

To import the user settings  and keyboard shortcuts (aka keybindings) used by me in general, import the settings.json and keybindings.json files to your user folder. This is usually [%AppData%/Roaming/code/User]. Adjust some paths which relate to my installation paths i.e. replace [c:\\Users\\giyash] in most cases. You can set specific keybindings in VS Code yourself as well. 

## List of software, most widely used within our group:
1. Python (programming)
2. Matlab (programming)
3. Texmaker + Miktex (Latex)
4. Overleaf (web Latex)
5. Inkscape / draw.io (Illustrations)
6. Spyder / Visual studio Code (IDE)
7. Planner / Mindmanager (Project management)
8. Citavi / Mendeley (reference manager)
9. FreeFileSync / FileZilla / WindSCP (sync managers)
10. Grammerly / DeepL (language translations)
11. Keepass (password manager)
12. Git (version controlling, use https://gitlab.cc-asp.fraunhofer.de)
13. Python environment manager  (most popular listed below)  
        -> Anaconda (big distribution)   
        -> Miniconda (without GUI, small distribution, dependency resolver but takes time)  
        -> Poetry (handles conda and pip, good dependency resolving)  
        -> pip (handles only pip packages, not a dependency resolver) 


## Data storage
1. Measurement campaign data is stored on `Z:/iwes.fraunhofer.de` (Add a network drive for the first time)
2. personal data on `oneDrive`
3. Additional data on `owncloud`
4. Measurement data GUI [oneDAS](https://onedas.iwes.fraunhofer.de/)


Contact us, if you have any questions on the above procedures:
- [Ashim Giyanani](https://teams.microsoft.com/l/chat/0/0?users=ashim.giyanani@iwes.fraunhofer.de) 
- [Paul Meyer](https://teams.microsoft.com/l/chat/0/0?users=paul.meyer@iwes.fraunhofer.de)



