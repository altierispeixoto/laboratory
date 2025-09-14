
conda create --name crdisney python=3.8

conda activate crdisney

conda install -c conda-forge poetry

poetry install
poetry build


sudo snap install heroku --classic

### Important commands
heroku apps


### buildpacks

[python-poetry-buildpack](https://elements.heroku.com/buildpacks/moneymeets/python-poetry-buildpack)


heroku buildpacks:clear
heroku buildpacks:add https://github.com/moneymeets/python-poetry-buildpack.git
heroku buildpacks:add heroku/python


heroku git:remote -a myapp
git push heroku main
