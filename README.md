# Set up 

## Download repository 

On your command line, do:

```
git clone git@github.com:Timothysit/oTreeExperiments.git
```

## Set up python environment 

1. Install UV, on linux/mac, run: `curl -LsSf https://astral.sh/uv/install.sh | sh`
2. Change to the repostiroy: `cd oTreeExperiments`
3. Then do `uv sync`

If you want to install `oTree` some other way, the installation instructions are here:

https://otree.readthedocs.io/en/latest/install.html


# Starting oTree and the experiment

Assuming you installed oTree using UV, change to the oTreeExperiemnts folder, and do:

```
uv run otree devserver
```

1. Click on the link provided 
2. Go to the Sessions tab
3. Click create new session and select `matching_pennies_solo_live`
4. Set number of participants to 2
5. Two links will be provided, one for each computer

