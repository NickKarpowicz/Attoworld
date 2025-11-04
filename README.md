# Attoworld

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15525871.svg)](https://doi.org/10.5281/zenodo.15525871)

Tools from the Attosecond science group at the Max Planck Institute of Quantum Optics, a.k.a. [Attoworld](https://www.attoworld.de)

**We are still in the building phase. Things may change, and haven't been validated; use at your own risk!**

[Documentation available here!](http://nickkarpowicz.github.io/docs/attoworld)

## Structure
The module has several submodules to keep it organized:
- *data*: classes and functions for handling the various data formats used across the labs
- *numeric*: numerical tools
- *personal*: a module where we can add our own functions that might not be of general interest, but are still good to have available to we can easily share work
- *plot*: functions for plotting with a consistent style
- *wave*: functions for processing waveforms and pulses
- *spectrum*: functions for processing spectra
- *attoworld_rs*: A place to put Rust code with a Python interface for cases where it's particularly important that the program be fast and correct.

## Contributing

### Code guidelines
The goal of this module is to gather the python programming work that we do, which maybe others in the group or the community at large could benefit from, into a module that we can easily add to our projects. This is easier if we follow some guidelines for best practices:
 - Use [docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) so that people know how to use your functions.
 - Comment code enough that it's understandable. It's possible that you write in a "self documenting" way, which is fine, but if you're doing something fancy and non-obvious, put in a note
 - If there's a function or class that you think others could benefit from, absolutely feel free to add it to the main modules. If you think you are likely the only one who will use something, you can also add a submodule to the attoworld.personal namespace. This makes it easier to share files with others!
 - If you're importing libraries, check and see if they're listed in the dependencies section of pyproject.toml - put them in if they're not there. This will help make sure that "pip install attoworld" will work out of the box.

 ### Working with git and Github
 I know a lot of you aren't familiar with git - it's incredibly powerful and useful, but has a learning curve. A lot of code editors have tools built-in for working with git, and these will do most of the work for you. However, it's important to set yourself up properly if you want to add code to the main repository with minimal difficulty.

 #### **Forking the repo and getting a local working copy**
 When you add or edit code here, it'll take place through a "pull request" (PR). This means that those of us maintaining the main repository have a chance to review and test changes before they're made, and that you can freely edit your version, using git for tracking changes, without affecting everyone else. Your edits will take place in what's called a fork. To create one, just click the "Fork" button at the top of this page. You'll have a chance to give it whatever name you like.

 Once you have your fork, it will be located at ```https://github.com/your_username/your_fork_name```. You can download this and start working with it using git from the command line:

 ```
 git clone https://github.com/your_username/your_fork_name
 ```

Now that you have your fork on your computer, you can work with it as you wish.

#### **Making a PR**
Once you've made your changes, to add them to the main repo, go to Github and make a pull request.

#### **After your PR**
Once your PR has been merged, it's a good idea to clean up your fork and synchronize with the main repo. Before you add new code, do this from the command line:

```
git remote add upstream https://github.com/NickKarpowicz/Attoworld
git fetch upstream
git checkout main
git reset --hard upstream/main
git push origin main --force
git fetch upstream --prune
```

Do this before you start a new wave of edits; it ensures that what you write is going to be compatible with the up-to-date version of the repo.

## Developing locally
To work with the repo, i.e. if you want to add functions and test them cleanly, here's how you can build and install it in a local virtual environment.

First, you should have [Maturin](https://github.com/PyO3/maturin) installed (e.g. ```pipx install maturin```). This is the build system used, which converts the repository into a usuable python module, and can install it for you.

Now, set up a [virtual environment](https://docs.python.org/3/library/venv.html). Navigate to wherever you want to create it and enter
```
python -m venv .venv
```
This will create a virtual environment in a folder named ".venv" (the dot at the front makes it hidden). You can enable this environment with

Linux/Mac:
```
source .venv/bin/activate
```
Windows:
```
.venv\Scripts\activate.bat
```
(or .\.venv\Scripts\Activate.ps1 in PowerShell).

Once your virtual environment is activated, you can build and install the package. This step is easy:

```
maturin develop
```

If it goes without errors, you'll have your working version of the package installed in a your virtual environment and can test with it. You will just have to run "maturin develop" once such that the compiled library is available, and now you can work. Changes made to the python code will affect the python module in the virtual environment.
