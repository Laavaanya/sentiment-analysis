from flask import Flask,render_template,request,redirect,url_for
import pandas as pd
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax
from statistics import mean
# import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Must come before importing pyplot
import matplotlib.pyplot as plt
import os
app=Flask(__name__)

@app.route("/",methods=["POST","GET"])
def home():
    return render_template("index.html")

@app.route("/user",methods=["POST","GET"])
def user():
    if request.method=="POST":
        if 'customer' in request.form:
            return redirect(url_for("review"))
        elif 'admin' in request.form:
            return redirect(url_for("login"))
        elif 'explore' in request.form:
            return redirect(url_for("dishRating"))
    return render_template("user.html")

@app.route("/login",methods=["POST","GET"])
def login():
    if request.method=="POST":
        if request.form['username']=='Fass25' and request.form['password']=='2509':
            return redirect(url_for("showPlots"))
        else:
            return render_template("loginFailed.html")
    else:
        return render_template("login.html")
    
@app.route("/review",methods=["POST","GET"])
def review():
    if request.method=="POST":
        dishName=request.form['dishName']
        review=request.form['review']
        data = {'dish_Names': [dishName],'Review': [review]}
        df = pd.DataFrame(data)
        df.to_csv(r'C:\Users\lavak\Downloads\test.csv', mode='a', index=False, header=False)
        return redirect(url_for("dish"))
    else:
        return render_template("userReview.html")

@app.route("/dish")
def dish():
    df = pd.read_csv(r'C:\Users\lavak\Downloads\test.csv')
    MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    config = AutoConfig.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    n = df.shape[0]
    for i in range(n):
        text = df['Review'][i]
        encoded_input = tokenizer(text, return_tensors='pt')
        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        ranking = np.argsort(scores)
        ranking = ranking[::-1]
        for k in range(3):
            ranking[k]=int(ranking[k])
            ranking[k] += 1
        df.loc[i, 'Rating'] = ranking[0]
    df.to_csv(r'C:\Users\lavak\Downloads\test.csv', index=False)
    return redirect(url_for("home"))

@app.route("/dishRating")
def dishRating():
    df = pd.read_csv(r'C:\Users\lavak\Downloads\test.csv')
    grouped_df = df.groupby('dish_names')['Rating'].apply(list).reset_index(name='reviews_per_dish')
    processed_data=[]
    for i in range(grouped_df.shape[0]):
        processed_data.append({'dishname': grouped_df['dish_names'][i], 'rating': mean(grouped_df['reviews_per_dish'][i])})
    print(processed_data)
    return render_template("dish.html",data=processed_data)

@app.route("/showPlots")
def showPlots():
    df = pd.read_csv(r'C:\Users\lavak\Downloads\test.csv')
    grouped_df = df.groupby('dish_names')['Rating'].apply(list).reset_index(name='reviews_per_dish')
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    processed_data=[]
    for i in range(int(grouped_df.shape[0])):
        processed_data.append(str(grouped_df['dish_names'][i]))
        rate=['1.0','2.0','3.0']
        no_of_customers=[]
        no_of_customers.append(grouped_df['reviews_per_dish'][i].count(1))
        no_of_customers.append(grouped_df['reviews_per_dish'][i].count(2))
        no_of_customers.append(grouped_df['reviews_per_dish'][i].count(3))
        fig = plt.figure(figsize = (5, 5))
        colors = ['yellow', 'lightgreen', 'skyblue']
        plt.bar(rate,no_of_customers,width = 0.2,color=colors)
        plt.xlabel("Rating")
        plt.ylabel("No_of_customers")
        plt.title(str(grouped_df['dish_names'][i]))
        plot_filename = os.path.join(app.root_path, 'static', f"{grouped_df['dish_names'][i]}.png")
        plt.savefig(plot_filename, format='png')
        plt.close() 
    return render_template("viewratings.html",data=processed_data)



@app.route('/submit', methods=['POST'])
def submit():
    df = pd.read_csv(r'C:\Users\lavak\Downloads\test.csv')
    grouped_df = df.groupby('dish_names')['Rating'].apply(list).reset_index(name='reviews_per_dish')
    if request.method == 'POST':               
        if 'Biryani' in request.form:
            data=['Biryani']
            data.append(grouped_df['reviews_per_dish'][0].count(1))
            data.append(grouped_df['reviews_per_dish'][0].count(2))
            data.append(grouped_df['reviews_per_dish'][0].count(3))
            return render_template('show_plots.html', img_data='Biryani.png',data=data)
        
        elif 'Dosa' in request.form:
            data=['Dosa']
            data.append(grouped_df['reviews_per_dish'][1].count(1))
            data.append(grouped_df['reviews_per_dish'][1].count(2))
            data.append(grouped_df['reviews_per_dish'][1].count(3))
            return render_template('show_plots.html', img_data='Dosa.png',data=data)
        
        elif 'Gobi65' in request.form:
            data=['Gobi65']
            data.append(grouped_df['reviews_per_dish'][2].count(1))
            data.append(grouped_df['reviews_per_dish'][2].count(2))
            data.append(grouped_df['reviews_per_dish'][2].count(3))
            return render_template('show_plots.html', img_data='Gobi65.png',data=data)
        
        elif 'Idly' in request.form:
            data=['Idly']
            data.append(grouped_df['reviews_per_dish'][3].count(1))
            data.append(grouped_df['reviews_per_dish'][3].count(2))
            data.append(grouped_df['reviews_per_dish'][3].count(3))
            return render_template('show_plots.html', img_data='Idly.png',data=data)
        
        elif 'Noodles' in request.form:
            data=['Noodles']
            data.append(grouped_df['reviews_per_dish'][4].count(1))
            data.append(grouped_df['reviews_per_dish'][4].count(2))
            data.append(grouped_df['reviews_per_dish'][4].count(3))
            return render_template('show_plots.html', img_data='Noodles.png',data=data)
        
        
        elif 'Parotta' in request.form:
            data=['Parotta']
            data.append(grouped_df['reviews_per_dish'][5].count(1))
            data.append(grouped_df['reviews_per_dish'][5].count(2))
            data.append(grouped_df['reviews_per_dish'][5].count(3))
            return render_template('show_plots.html', img_data='Parotta.png',data=data)
        
        elif 'Pasta' in request.form:
            data=['Pasta']
            data.append(grouped_df['reviews_per_dish'][6].count(1))
            data.append(grouped_df['reviews_per_dish'][6].count(2))
            data.append(grouped_df['reviews_per_dish'][6].count(3))
            return render_template('show_plots.html', img_data='Pasta.png',data=data)
       
        elif 'Pizza' in request.form:
            data=['Pizza']
            data.append(grouped_df['reviews_per_dish'][7].count(1))
            data.append(grouped_df['reviews_per_dish'][7].count(2))
            data.append(grouped_df['reviews_per_dish'][7].count(3))
            return render_template('show_plots.html', img_data='Pizza.png',data=data)
        
        elif 'Pongal' in request.form:
            data=['Pongal']
            data.append(grouped_df['reviews_per_dish'][8].count(1))
            data.append(grouped_df['reviews_per_dish'][8].count(2))
            data.append(grouped_df['reviews_per_dish'][8].count(3))
            return render_template('show_plots.html', img_data='Pongal.png',data=data)
        
        elif 'Poori' in request.form:
            data=['Poori']
            data.append(grouped_df['reviews_per_dish'][9].count(1))
            data.append(grouped_df['reviews_per_dish'][9].count(2))
            data.append(grouped_df['reviews_per_dish'][9].count(3))
            return render_template('show_plots.html', img_data='Poori.png',data=data)
        
        elif 'Samosa' in request.form:
            data=['Samosa']
            data.append(grouped_df['reviews_per_dish'][10].count(1))
            data.append(grouped_df['reviews_per_dish'][10].count(2))
            data.append(grouped_df['reviews_per_dish'][10].count(3))
            return render_template('show_plots.html', img_data='Samosa.png',data=data)
        
        elif 'Vada' in request.form:
            data=['Vada']
            data.append(grouped_df['reviews_per_dish'][11].count(1))
            data.append(grouped_df['reviews_per_dish'][11].count(2))
            data.append(grouped_df['reviews_per_dish'][11].count(3))
            return render_template('show_plots.html', img_data='Vada.png',data=data)


@app.route('/negativeReviews', methods=['POST'])
def negativeReviews():
    df = pd.read_csv(r'C:\Users\lavak\Downloads\test.csv')
    m=df.shape[0]
    for i in range(m):
        if(df['Rating'][i]==1.0):
            data1 = {'dish_Names':[df['dish_names'][i]],'Review':[df['Review'][i]]}
            df1 = pd.DataFrame(data1)
            df1.to_csv(r'C:\Users\lavak\Downloads\negativeReviews.csv', mode='a', index=False, header=False)
    df1 = pd.read_csv(r'C:\Users\lavak\Downloads\negativeReviews.csv')
    grouped_df1 = df1.groupby('dish_names')['Review'].apply(list).reset_index(name='reviews_per_dish')
    if request.method=='POST':
        for i in range(grouped_df1.shape[0]):
            
            if grouped_df1['dish_names'][i] in request.form:
                dishname=grouped_df1['dish_names'][i]
                a=[]
                for j in range(len(grouped_df1['reviews_per_dish'][i])):
                    a.append(grouped_df1['reviews_per_dish'][i][j])
                b=list(set(a))

                return render_template("negativeReviews.html",data=b,name=dishname)
        
if __name__=="__main__":
    app.run()