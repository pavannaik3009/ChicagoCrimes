import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

df = pd.read_csv("CrimesChicago.csv", delimiter=',', index_col=0, iterator=True)
crime_df = pd.concat(df, ignore_index=True).reset_index()

crime_df.shape

crime_df.info()

crime_df.head()

crime_df.isnull().sum()

#Visuals of columns and size of missing data
plt.figure(figsize=(10,7))
sns.heatmap(crime_df.isnull(), cbar = False, cmap = 'Blues')

#Data Munging and Attribute correlation
crime_df.Date = pd.to_datetime(crime_df.Date, format='%m/%d/%Y %I:%M:%S %p')
crime_df.index = pd.DatetimeIndex(crime_df.Date)
crime_df=crime_df[pd.notnull(crime_df['Location Description'])]
crime_df=crime_df[pd.notnull(crime_df['District'])]
crime_df.drop(['Ward', 'Community Area', 'Latitude', 'Longitude', 'Location'], axis=1, inplace=True)
corr = crime_df.corr()
fig, ax = plt.subplots(figsize=(10,10))  
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);

#2

#Number of crimes by months of the year
plt.figure(figsize=(8,8))
crime_df.groupby(crime_df.index.month).size().plot(kind='bar', legend=False, color='#B0C4DE')
ax = plt.subplot()
plt.xlabel('Months of the year')
plt.ylabel('Number of crimes')
plt.title('Number of crimes by month of the year')
mon = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
ax.set_xticklabels(mon)
plt.show()

#Number of crimes by days of the week
plt.figure(figsize=(8,8))
crime_df.groupby([crime_df.index.dayofweek]).size().plot(kind='bar', color='#CD5C5C')
plt.xlabel('Days of the week')
days = ['Mon','Tue','Wed',  'Thu', 'Fri', 'Sat', 'Sun']
plt.xticks(np.arange(7), days)
plt.ylabel('Number of crimes')
plt.title('Number of crimes by day of the week')
plt.show()


#3

#Crimes over the years
plt.figure(figsize=(10,5))
crime_df.resample('M').size().plot(legend=False, color='Red')
plt.title('Number of crimes per month over the years')
plt.xlabel('Years')
plt.ylabel('Number of crimes')
plt.show()



#4

#Crimes vs arrests
#No of arrests and non arrests
crime_df['Arrest'].value_counts()
arrest_true=crime_df[crime_df['Arrest']==1]['Primary Type'].value_counts().to_frame()
arrest_true
primary_type=crime_df[crime_df['Arrest']==1]['Primary Type'].sort_values(ascending=False).value_counts().index.tolist()
arrest_true=crime_df[crime_df['Arrest']==1]['Primary Type'].value_counts().tolist()
plt.figure(figsize=(10,10))
plt.barh(primary_type,arrest_true,height=0.6,color ='#E9A000')
plt.title('Arrests per primary type')
plt.xlabel('Arrests')
plt.ylabel('Primary Type')
plt.yticks(primary_type,fontsize=10)
plt.show()

#5

#Location with High Crime rate
data = crime_df.loc[(crime_df['X Coordinate']!=0)]
sns.lmplot('X Coordinate', 
           'Y Coordinate',
           data=data[:],
           fit_reg=False, 
           hue="District",
           palette='pastel',
           height=12,
           ci=2,
           scatter_kws={"marker": "D", 
                        "s": 10})
ax = plt.gca()
ax.set_title("Crime Distribution per District")
crime_location = pd.DataFrame(crime_df.groupby('Location Description').size().sort_values(ascending=False).rename('Count').reset_index())
crime_location.head()
plt.figure(figsize=(7,7))
plt.bar(crime_df['Location Description'].value_counts().index.tolist()[0:15],\
        crime_df['Location Description'].value_counts().tolist()[0:15],color = '#9C2706')
plt.title('Crime by location')
plt.xlabel('Location')
plt.ylabel('Number of crimes')
plt.xticks(crime_df['Location Description'].value_counts().index.tolist()[0:15],rotation=90)
plt.show()


#6

#Percentage of Domestic crimes that ended an arrest

domarr_false, domarr_true = crime_df[crime_df['Domestic']==1]['Arrest'].value_counts()
dom_false, dom_true = crime_df['Domestic'].value_counts()
arr_false, arr_true = crime_df['Arrest'].value_counts()
sns.countplot(x='Arrest',data=crime_df)
plt.ylabel('No of Crimes')
plt.show()
domarr_true/(dom_true+arr_true)*100


#7

#Number of arrests per crime group

crimeGroups = {'NARCOTICS' : '3', 'BATTERY' : '1', 'THEFT':'2', 'WEAPONS VIOLATION': '5', 'CRIMINAL TRESPASS' : '2',
'OTHER OFFENSE' : '4','ASSAULT' : '1','CRIMINAL DAMAGE' : '2','INTERFERENCE WITH PUBLIC OFFICER' : '1',
'PUBLIC PEACE VIOLATION' : '1', 'DECEPTIVE PRACTICE' : '3','ROBBERY' : '2','PROSTITUTION' : '3',
'BURGLARY' : '2','MOTOR VEHICLE THEFT' : '3','OFFENSE INVOLVING CHILDREN' : '3',
'LIQUOR LAW VIOLATION' : '3','CONCEALED CARRY LICENSE VIOLATION' : '5',
'GAMBLING' : '3','SEX OFFENSE' : '1','HOMICIDE' : '1','CRIM SEXUAL ASSAULT' : '1',
'OBSCENITY' : '1','ARSON' : '2','STALKING' : '1','PUBLIC INDECENCY' : '3',
'INTIMIDATION' : '1', 'DOMESTIC VIOLENCE' : '1','KIDNAPPING': '1', 'NON-CRIMINAL (SUBJECT SPECIFIED)':'6',
'OTHER NARCOTIC VIOLATION' : '3', 'NON - CRIMINAL' : '6', 'RITUALISM' : '7', 'HUMAN TRAFFICKING' : '1',
              'NON-CRIMINAL' : '6'}
crime_df['crimeGroups'] = crime_df['Primary Type'].apply(lambda x : crimeGroups[x])
cri_gro=crime_df[crime_df['Arrest'] == 0 ]['crimeGroups'].value_counts().sort_index().index.tolist()
arr_false=crime_df[crime_df['Arrest'] == 0]['crimeGroups'].value_counts().sort_index().tolist()
plt.figure(figsize=(7,7))
plt.bar(cri_gro,arr_false, color = '#9C2706')
plt.title('Number of Arrest per crime groups')
plt.xlabel('Crime Groups')
plt.ylabel('Number of arrests')
plt.show()

#Number of crimes in each district
Crime_By_Dist= crime_df.pivot_table('Arrest', aggfunc = np.size, columns = 'District',
                               index = crime_df.index.year, fill_value = 0)
plt.figure(figsize = (20,8))
plt.title('Number of Crimes By Year In Each District')
hm = sns.heatmap(Crime_By_Zip, cmap = 'YlOrRd', linewidth = 0.01, linecolor = 'k')

#Different crimes vs hours of the day
hourly_crimes = crime_df.pivot_table(values='index', index='Primary Type',columns=crime_df['Date'].dt.hour, aggfunc=np.size)
plt.figure(figsize = (12,10))
sns.heatmap(hourly_crimes)
plt.xlabel('Hours of the day')
plt.ylabel('Type of Crime')
plt.show()


