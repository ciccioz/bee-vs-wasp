# bee-vs-wasp
end to end DL project with image recognition module

- inspiration post: https://towardsdatascience.com/how-to-build-and-deploy-a-machine-learning-web-application-in-a-day-f194fdbd4a5f
- Data taken from: https://www.kaggle.com/jerzydziewierz/bee-vs-wasp
- First run: after data download, launch Python Notebook in order to get the correct data structure used as input for model training.
- Training has been done using Xception model, model output saved in file `xception.h5`.

## Run code before app
- get data from this link: https://www.kaggle.com/jerzydziewierz/bee-vs-wasp
- run `bee_vs_wasp.ipynb` changing path of your working directory (tip: launch everything in Google Colab)

## Run app
- in the terminal, run command `streamlit run app.py` 

## Deployment

In repo folder:

```heroku login
git add .
git commit -m "Enter your message here"
git push heroku master
heroku ps:scale web=1```
