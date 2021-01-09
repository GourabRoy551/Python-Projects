#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Setup libraries


# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import chart_studio as pl
import plotly.offline as of
import cufflinks as cf
import datetime as dt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


of.init_notebook_mode(connected = True)
cf.go_offline()


# In[5]:


# Load Datafiles


# In[6]:


donations = pd.read_csv('Donations.csv')


# In[7]:


donors = pd.read_csv('Donors.csv')


# In[8]:


projects = pd.read_csv('Projects.csv')


# In[9]:


resources = pd.read_csv('Resources.csv')


# In[10]:


schools = pd.read_csv('Schools.csv')


# In[11]:


teachers = pd.read_csv('Teachers.csv')


# In[12]:


#Describe and show data for column ideas


# In[13]:


print('Shape of donations dataframe is:' , donations.shape)
print('Shape of donors dataframe is:' , donors.shape)
print('Shape of projects dataframe is:' , projects.shape)
print('Shape of resources dataframe is:' , resources.shape)
print('Shape of schools dataframe is:' , schools.shape)
print('Shape of teachers dataframe is:' , teachers.shape)


# In[14]:


donations.head()


# In[15]:


donors.head()


# In[16]:


projects.head()


# In[17]:


resources.head()


# In[18]:


schools.head()


# In[19]:


donations.describe()


# In[ ]:


data = pd.merge(donations , projects , how='inner' , on = 'Project ID')


# In[ ]:


data2 = pd.merge(data , donors , how='inner' , on='Donor ID')


# In[ ]:


data3 = pd.merge(data2 , schools , how='inner' , on='School ID')


# In[ ]:


data4 = pd.merge(data3, teachers , how='inner' , on='Teacher ID')


# In[ ]:


data4.head()


# In[ ]:


a = data4.columns.values.tolist()
a


# In[ ]:


#Which 10 states have the most number of schools that opened projects to gather donations ? Plot the data using bar plot.


# In[ ]:


s = schools['School State'].value_counts().sort_values(ascending = False).head(10)
s


# In[ ]:


s.iplot(kind='bar' , xTitle='States' , yTitle='Number of schools' , title='Number of schools involved in projects by states')


# In[ ]:


s2 = data4.groupby('School State')['Donation Amount'].mean().sort_values(ascending=False).head(10)
s2


# In[ ]:


s2.iplot(kind='bar' , xTitle='State' , yTitle='Average donation per project' 
         , title='Top 10 states(with maximum doantion)' , colorscale='paired' )


# In[ ]:


mean = np.mean(data4['Donation Amount'].dropna())
median = np.median(data4['Donation Amount'].dropna())
percentiles = np.percentile(data4['Donation Amount'].dropna() ,[25,75])
minimum = data4['Donation Amount'].dropna().min()
maximum = data4['Donation Amount'].dropna().max()

print('mean donation amount is:' ,np.round(mean,2))
print('median donation amount is:' ,median)
print('25% and 75% donation amount is:' ,percentiles)
print('minimum donation amount is:' ,minimum)
print('maximum donation amount is:' ,maximum)


# In[ ]:


x = np.sort(data4["Donation Amount"].dropna())
y = np.arange(1,len(x)+1)/len(x)
plt.plot(x,y,marker = '.')


# In[ ]:


s3 = data4.groupby('Donor State')['Donation ID'].count().sort_values(ascending = False).head(15)
s3


# In[ ]:


s4 = schools['School State'].value_counts()
s5 = data4.groupby('Donor State')['Donation ID'].count()
df = pd.concat([s4,s5],axis=1,keys=['Projects','Donations'])


# In[ ]:


df = df.dropna()


# In[ ]:


df.head()


# In[ ]:


df.iplot(kind='scatter',xTitle='Projects',
         yTitle='Donations',title='Projects vs Donations',
         symbol='x',colorscale='paired',mode='markers')


# In[ ]:


slope,intercept = np.polyfit(df.Projects,df.Donations,1)
x = np.array([df.Projects.min(),df.Projects.max()])
y = slope*x + intercept
plt.plot(x,y)


# In[ ]:


df.plot.scatter(x='Projects' , y='Donations')
slope,intercept = np.polyfit(df.Projects,df.Donations,1)
x = np.array([df.Projects.min(),df.Projects.max()])
y = slope*x + intercept
plt.plot(x,y)
plt.tight_layout()
plt.margins(0.05)


# In[ ]:


data4.head(2)


# In[ ]:


s6 = data4["Project Type"].value_counts()
s6


# In[ ]:


s7 = data4.groupby('Project Type')['Donation Amount'].sum().astype(int)
s7


# In[ ]:


plt.subplot(2,1,1)
plt.pie(s6 , startangle=90)
plt.subplot(2,1,2)
plt.pie(s7 , startangle=90)
plt.tight_layout()
plt.margins(0.05)
fig = plt.gcf()
fig.set_size_inches(25,15)


# In[ ]:


data4['Project Subject Category Tree'].nunique()


# In[ ]:


s8 = data4.groupby('Project Subject Category Tree')['Donation Amount'].sum().astype(int).sort_values(ascending = False).head(15)
s8


# In[ ]:


s9 = s8/1000000
s9.iplot(kind="bar" , xTitle='Project sub category' , yTitle='Donation amount in millions',
        title='Donation amount by project subject' , colorscale='paired')


# In[ ]:


data4[['Project Posted Date' , 'Project Fully Funded Date']].isnull().sum()


# In[ ]:


data4[['Project Posted Date' , 'Project Fully Funded Date']].head()


# In[ ]:


data4['Project Posted Date'] = pd.to_datetime(data4['Project Posted Date'])


# In[ ]:


data4['Project Fully Funded Date'] = pd.to_datetime(data4['Project Fully Funded Date'])


# In[ ]:


data4['Funding Time'] = data4['Project Fully Funded Date'] - data4['Project Posted Date'] 
data4[['Funding Time','Project Posted Date' , 'Project Fully Funded Date']].head()


# In[ ]:


data4[['Funding Time','Project Posted Date' , 'Project Fully Funded Date']].isnull().sum()


# In[ ]:


data5 = data4[pd.notnull(data4['Funding Time'])]
data5[['Funding Time','Project Posted Date' , 'Project Fully Funded Date']].isnull().sum()


# In[ ]:


import datetime as dt
data5['Funding Time'] = data5['Funding Time'].dt.days


# In[ ]:


data5[['Funding Time','Project Posted Date' , 'Project Fully Funded Date']].head()


# In[ ]:


wrong_overall_mean_time = data5['Funding Time'].mean()
wrong_overall_mean_time


# In[ ]:


overall_mean_time = data5.groupby('Project ID')['Funding Time'].mean()
output = overall_mean_time.mean()
output


# In[ ]:


#Average funding time for each state

state_project_funding_time = data5.groupby(['School State' , 'Project ID'])['Funding Time'].mean()
state_project_funding_time


# In[ ]:


state_average_project_funding_time = state_project_funding_time.groupby('School State').mean()
state_average_project_funding_time.round(0)


# In[ ]:


fast = state_average_project_funding_time.round(0)
fast[fast<32].sort_values().head(10)


# In[ ]:


fast_funding = fast[fast<32].sort_values().head(10)
fast_funding.iplot(kind='bar' , xTitle='States' , yTitle='fully funding time(in days)',
                  title='states that fund projects earlier than others',
                  colorscale='paired')


# In[ ]:


slow = state_average_project_funding_time.round(0)
slow[slow>32].sort_values(ascending = False).head(10)


# In[ ]:


slow_funding = slow[slow>32].sort_values(ascending = False).head(10)
slow_funding.iplot(kind='bar' , xTitle='States' , yTitle='fully funding time(in days)',
                  title='states that fund projects earlier than others'
                  )

