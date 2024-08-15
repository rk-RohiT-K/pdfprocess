#Imports
import re
import os
import nltk
import warnings
#Warning Sup
warnings.filterwarnings("ignore", category=DeprecationWarning)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF
from nltk.tokenize import word_tokenize
from tabula.io import read_pdf
from datetime import datetime, timedelta

#class for storing month's information
class month_data:
    def __init__(self,month_frame,month_name):
        self.month_frame = month_frame
        self.month_name = month_name
        self.starting_balance = month_frame['Closing'].values[-1]
        self.closing_balance = month_frame['Closing'].values[0]
        self.total_credit = np.sum(month_frame['Deposit'].values)
        self.total_debit = np.sum(month_frame['Withdrawal'].values)

#GLOBL Variables
excel_path = 'sub1.xlsx'
today_date = datetime.today()
report_path = 'report_' + today_date.strftime('%d_%b_%Y') + '.pdf'
img_path = 'images/'
columns = ['Date','Narration','ref','Value','Withdrawal','Deposit','Closing']
category_frame = pd.read_csv('category.csv') if os.path.exists('category.csv') else pd.DataFrame({'Category':[],'Tag':[]})

def make_frame_excel(excel_path,columns):
    df = pd.read_excel(excel_path)
    df.columns = columns
    df = df.drop(columns = ['ref','Value'])
    df = df.fillna(0)

    # Processing Date to required format
    df['Date'] = df['Date'].astype(str).apply(lambda x : datetime.strptime(x,'%d/%m/%y').strftime('%d %b %Y'))

    # Processing Narration
    narr = df['Narration'].astype(str).values
    def process(s):
        s = s.replace(' ','_')
        s = s.replace('-',' ')
        s = s.replace('.',' ')
        return s
    narr = np.array([process(s) for s in narr])
    tokenized_strings = [word_tokenize(s) for s in narr]
    def has_numbers(string):
        return bool(re.search(r'\d', string))
    def process_tokens(tokenized_strings):
        arr = []
        for tokens in tokenized_strings:
            if tokens[0] == 'UPI':
                arr.append(tokens[1])
            else:
                narr = ''
                for word in tokens:
                    if '_' in word:
                        narr += word + '_'
                        break
                    elif has_numbers(word):
                        break
                    narr += word + '_'
                arr.append(narr)
        return np.array(arr)
    df['Narration'] = process_tokens(tokenized_strings)
    print("DataFrame generated and processed....")
    return df

#The Tags
def add_tags(data_frame,category_frame,column='Narration',tag_column='Tags'):
    if data_frame[column].dtype != 'O':
        print("Wrong Column")
        return
    tags = []
    for narr in data_frame[column].astype(str).apply(lambda x: x.lower()):
        category_found = False
        for category,tag in zip(category_frame['Category'].astype(str),category_frame['Tag'].astype(str)):
            if narr.__contains__(category):
                tags.append(tag)
                category_found = True
                break
        if not category_found:   
            print(f"New Type : '{narr}' detected, Classify (category in all small)---")
            category = input("Category : ") or narr
            tag = input("Tag : ") or 'Person'
            category_frame.loc[len(category_frame.index)] = [category,tag]
            tags.append(tag)
            print("Added!!")
    data_frame[tag_column] = tags
    print("Added Tags successfully...")
    category_frame.to_csv('category.csv',index=False)
    return

#The dictionary of periods
def segragate(df):
    dates = df['Date'].astype(str)
    years = np.unique(dates.apply(lambda x : int(x[-5:])).values)
    df['years'] = dates.apply(lambda x : int(x[-5:])).values
    gr_yr = df.groupby('years')
    year_frames = {group_key: group.reset_index(drop=True).drop('years',axis=1) for group_key, group in gr_yr}
    for year, frame in year_frames.items():
        dates = frame['Date'].astype(str)
        frame['months'] = dates.apply(lambda x : x[3:-5]).values
        gr_mnth = frame.groupby('months',sort=False)
        mnth_frames = {group_key: month_data(group.reset_index(drop=True).drop('months',axis=1),group_key) for group_key, group in gr_mnth}
        year_frames[year] = mnth_frames
    print('Generated Dictionary....')
    year_frames[2024]['Feb'].month_frame.to_csv('feb_frame.csv')
    return year_frames

#The Charts
def pie_chart(df,column,img_name):
    data_ = df[column].values
    tags = df['Tags'].astype(str).values
    labels = np.unique(tags)
    data = {}
    for label in labels:
        sum_ = np.sum(data_[tags == label])
        if sum_ != 0:
            data[label] = sum_
    labels = list(data.keys())
    data = list(data.values())
    fig, ax = plt.subplots(figsize=(10,7))
    total = sum(data)
    percentages = [size / total for size in data]
    explode = [0.1*pct for pct in percentages]
    colors = plt.cm.Greys(np.linspace(0.1, 0.8, len(labels)))

    def f(pct, allvalues):
        val = pct / 100.*np.sum(allvalues)
        return '{:.1f}%\n{:.2f} INR'.format(pct, val)
    wedges, texts, autotexts = ax.pie(data,
                                      autopct=lambda pct: f(pct,data),
                                      explode=explode,
                                      labels=labels,
                                      colors=colors,
                                      shadow=False,
                                      textprops={'color': 'black', 'fontsize': 14,'fontfamily':'serif','rotation':30},
                                      pctdistance=0.95)
    ax.legend(wedges, labels, title=column,loc = "center left",bbox_to_anchor=(1,0,0.5,1))
    plt.setp(autotexts,size=9,weight='bold')
    for autopct in autotexts:
        autopct.set_color('red')
    ax.set_title(img_name)
    plt.savefig(img_path + img_name+'_pie.jpeg',dpi=300,bbox_inches='tight')
    plt.clf()

#The Line Chart
def linegraph(df,column,img_name):
    pre_data = df[column].values
    pre_dates = df['Date'].astype(str).values

    unique_dates = np.unique(pre_dates)
    summed_data = []
    for date in unique_dates:
        summed_data.append(np.sum(pre_data[pre_dates == date]))
    summed_data = np.array(summed_data)
    start_date = pre_dates[0]
    end_date = pre_dates[-1]
    dates = generate_date_range(start_date,end_date)
    data = np.zeros_like(dates,dtype=float)
    for i in range(len(dates)):
        if dates[i] in unique_dates:
            data[i] = summed_data[unique_dates == dates[i]]
    dates = [date[:-5] for date in dates]
    plt.scatter(dates,data,color='gray')
    plt.plot(dates, data, color='brown', linestyle='-', linewidth=0.2)
    for label, d in zip(dates, data):
        plt.vlines(label, 0, d, colors='gray', linewidth=1)
        if d:
            plt.text(label, d+2, f'{d:.2f}', ha='center', va='bottom', fontdict={'family': 'serif', 'size': 10})
    plt.xticks(rotation=-60, ha='right')
    plt.title(img_name)
    plt.savefig(img_path + img_name + '_line.jpeg',dpi=300,bbox_inches='tight')
    plt.clf()

#Helper Function
def generate_date_range(start_date, end_date):
    # Convert start_date and end_date strings to datetime objects
    start = datetime.strptime(start_date, '%d %b %Y')
    end = datetime.strptime(end_date, '%d %b %Y')

    # Initialize an empty list to store formatted dates
    formatted_dates = []

    # Iterate through the date range and format each date
    current_date = start
    while current_date <= end:
        formatted_date = current_date.strftime('%d %b %Y')  # Format date as "DD {First three letters of month} YYYY"
        formatted_dates.append(formatted_date)
        current_date += timedelta(days=1)  # Increment current_date by one day

    return formatted_dates

def generate_images(frames):
    for year,mnth_frames in frames.items():
        for mnth,mnth_data in mnth_frames.items():
            columns = ['Withdrawal','Deposit']
            for column in columns:
                pie_chart(mnth_data.month_frame,column,column+'_'+mnth+'_'+str(year))
                linegraph(mnth_data.month_frame,column,column+'_'+mnth+'_'+str(year))
    print('Generated Images...')

if __name__ == '__main__':
    df = make_frame_excel(excel_path,columns)
    add_tags(df,category_frame)
    frames_split = segragate(df)
    generate_images(frames_split)

    #Generating report in pdf
    height = 297 - 20
    width = 210 - 20 
    pdf = FPDF()
    pdf.set_auto_page_break(False)

    #====================TITLE_PAGE===============================
    pdf.add_page()
    h = height
    pdf.set_font("Arial",'BU',size=32)
    pdf.set_y(h / 2)
    pdf.cell(200, 10, txt="Financial Report", ln=True, align='C')
    pdf.set_font("Arial", size=20)
    pdf.set_y(h / 2 + 20)  # Adjust the value as needed
    start_year = list(frames_split.keys())[0]
    end_year = list(frames_split.keys())[-1]
    if start_year == end_year:
        subtitle = str(start_year)
    else:
        subtitle = str(start_year)+'-'+str(end_year)
    pdf.cell(200, 10, txt=subtitle , ln=True, align='C')
    pdf.set_font("Arial", size=10)
    pdf.set_y(h)  # Adjust the value as needed
    pdf.cell(0, 5, txt="Rohit Kumar", ln=True, align='R')
    pdf.cell(0, 5, txt=today_date.strftime('%d %B %Y'), ln=True, align='R')

    #===================YEAR_WISE_REPORT==========================
    dict_mnth = {month[:3]: month for month in [
        "January", "February", "March", "April", "May", "June", 
        "July", "August", "September", "October", "November", "December"
    ]}
    pdf.set_auto_page_break(True,margin=1.0)
    image_height = 75
    for year, mnth_frame in frames_split.items():
        pdf.add_page()
        pdf.set_font("Arial",'B',20)
        pdf.cell(0,10,txt=str(year),border=1,ln=1, align='C')
        for mnth, frame in mnth_frame.items():
            pdf.set_font("Arial",'B',16)
            pdf.cell(0,10,txt=dict_mnth[mnth],border = 'B',ln=1, align='C')
            pdf.set_font("Arial",'',10)
            pdf.cell(0,10,txt=f'Starting Balance : {frame.starting_balance} \t Closing Balance : {frame.closing_balance}',ln=1,align='L')
            categories = {'Withdrawal':frame.total_debit,'Deposit':frame.total_credit}

            charts = ['pie','line']
            for category,amount in categories.items():
                pdf.cell(0,5,txt=category + ' : ' + str(amount),ln=1,align='L')
                y = pdf.get_y()
                offset = 5
                for chart in charts:
                    img_name = img_path + category + '_' + mnth + '_' + str(year) + '_' + chart + '.jpeg'
                    pdf.image(img_name,offset,y,width/2-1)
                    offset += width/2
                pdf.set_y(y + image_height)
    #=================OUTPUT=======================================
    pdf.output(report_path)
    print('Report Made....')
            