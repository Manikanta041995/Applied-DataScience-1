#importing required documents
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import random


def read_file(path):
    """This function will take path as a parameter
       and reads the csv file and returns a Data Frame."""
    # Read the csv file
    books_df = pd.read_csv(filepath_or_buffer=path, sep=',', skip_blank_lines=True)
    return books_df


def info(df):
    """This function takes data frame as input and returns
       structure of the data frame such as columns,head,tail
       ,transpose,summary"""
    print('Columns of the Data Frame\n')
    print(df.columns)
    print('\n\n')
    print('The top values of Data Frame\n')
    print(df.head())
    print('\n\n')
    print('The bottom values of Data Frame\n')
    print(df.tail())
    print('\n\n')
    print(f'The size of the data frame : {df.size}\n')
    print(f'The shape of the data frame : {df.shape}\n')
    print('The transpose of Data Frame\n')
    print(df.T)
    print('\n\n')
    print('summary of the Data Frame\n')
    print(df.info(verbose = True))
    print('\n\n')



def preprocessing(df):
    """Handles missing values, deletes unnecessary columns, 
    and converts 'Price' to float."""
    missing_values = df.isnull().sum()
    print(missing_values)
    #Deleting rows with null values
    df = df.dropna()
    df.drop(columns = ['Unnamed: 0', 'URLs'], inplace=True)
    # Remove currency symbol from 'Price' and convert to float
    df['Price'] = df['Price'].str.replace('₹', '').str.replace(',', '').astype(float)
    return df



def analysis(df):
    """Performs a brief analysis of numerical columns and 
    returns kurtosis and skewness."""
    print('Brief analysis of Numerical Columns')
    print(df.describe())
    #calucating the skewness
    skew = df.skew(numeric_only = True)
    print('\n')
    print('The skewness of Numerical Columns')
    print(skew)
    #calucating the kurtosis
    kurt = df.kurtosis(numeric_only = True)
    print('\n')
    print('The kurtosis of Numerical Columns')
    print(kurt)
    return [kurt, skew]



def hist_plot(df, image_name):
    """Plots histograms for 'Price', 'Rating', and 'No. of People rated'."""
    #setting the backgroud to whitgrid.
    sns.set_style("whitegrid")
    plt.figure(figsize=(18, 14), dpi=200)
    #creating 3 subplots in frist column.
    plt.subplot(3, 1, 1)
    #ploting histogram in frist row for price
    sns.histplot(df['Price'], bins=50, kde=True)
    plt.ylabel('Frequency')
    plt.title('Distribution of Price')
    plt.subplot(3, 1, 2)
    #ploting histogram in second row for Rating
    sns.histplot(df['Rating'], bins=30, kde=True)
    plt.ylabel('Frequency')
    plt.title('Distribution of Rating')
    plt.subplot(3, 1, 3)
    #ploting histogram in Third row for No of people rated
    sns.histplot(df['No. of People rated'], bins=50, kde=True)
    plt.ylabel('Frequency')
    plt.title('Distribution of No. of People rated')
    plt.tight_layout()
    plt.savefig(image_name,dpi='figure',bbox_inches='tight')
    plt.show()


def colours():
    """Generates a list of colors for plotting."""
    # taking a list of colors for matplotlib into a list
    col = list(mcolors.CSS4_COLORS.keys())
    col1 = []
    for c in col:
        # keeping only dark and medium colors ignoring grey
        if 'dark' in c or 'medium' in c and 'grey' not in c:
            col1.append(c)
    colours = []
    for n in random.sample(range(0,28), 28):
        colours.append(col1[n])
    #returning a list of colors of size 28.
    return colours


def bar_chart(x, y, color, xlabel, ylabel, r, title, image_name):
    """Creates a bar chart."""
    plt.figure()
    # Create the bar chart with specified x and y values
    bar = plt.bar(x,y)
    # Set colors for each bar using the provided color list
    for n in range(len(color)):
        bar[n].set_color(color[n])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=r)
    plt.title(title)
    # Set x-axis margins to improve visual appearance
    plt.margins(x=0)
    plt.savefig(image_name,dpi='figure',bbox_inches='tight')
    plt.show()



def heatmap(df, image_name):
    """Plots a heatmap for the correlation between numeric columns."""
    plt.figure()
    # Creating Heatmap for Numeric columns
    sns.heatmap(df.corr(),annot=True,vmin=-1, vmax=1, 
                 annot_kws={'fontsize':8, 'fontweight':'bold'})
    plt.title('Correlation heatmap between numeric columns')
    plt.savefig(image_name,dpi='figure',bbox_inches='tight')
    plt.show()




def scatter_plot(df, image_name):
    """Creates a scatter plot for 'Price' vs 'Rating Points'."""
    plt.figure()
    # Scatter plot with 'Price' on the x-axis and 'Point' on the y-axis
    plt.scatter(df['Price'],df['Point'])
    plt.xlabel('Price')
    plt.ylabel('Rating Points')
    plt.title('Price vs Rating of Books')
    # Set x-axis scale to logarithmic for better visualization
    plt.xscale('log')
    plt.savefig(image_name,dpi='figure',bbox_inches='tight')
    plt.show()




def box_plot(df, image_name):
    """Creates a box plot for outliers in numeric columns."""
    columns = ['Price', 'Rating', 'No. of People rated']
    plt.figure()
    # Generate a box plot for the specified columns in the DataFrame
    plt.boxplot(df[columns])
    # Set y-axis scale to logarithmic for better visualization
    plt.yscale('log')
    # Set x-axis ticks and labels for each column
    plt.xticks([1,2,3],columns)
    plt.ylabel('Frequency')
    plt.title('Box plot for outliers in Numeric cols')
    plt.savefig(image_name,dpi='figure',bbox_inches='tight')
    plt.show()



# Specify the path to the CSV file
path = "Books_df.csv"

# Read data from the CSV file and store it in a DataFrame
df = read_file(path)

# Display information about the DataFrame (columns, head, tail, etc.)
info(df)

# Perform data preprocessing to handle missing values and convert 'Price' to float
df = preprocessing(df)

# Display information about the DataFrame after preprocessing
info(df)

# Perform a brief analysis of numerical columns and get skewness and kurtosis values
analysis(df)
analysis_values = analysis(df)

# Generate a heatmap to visualize the correlation between numeric columns
heatmap(df, 'Heatmap for Numeric cols')

# Generate histograms for the distribution of 'Price', 'Rating', and 'No. of People rated'
hist_plot(df, 'Histplot for distribution')

# Generate a box plot to identify outliers in numeric columns
box_plot(df, 'Boxplot for outliers')

# Group by 'Main Genre' and calculate the average price, then create a bar chart
genre_price = df.groupby("Main Genre")[["Price"]].mean().reset_index()
color = colours()
bar_chart(genre_price['Main Genre'], genre_price['Price'], color, 'Main Genre',
          'Price', 90, 'Average price by Main Genre', 
          'Bar chart for average price by Genre')

# Calculate a new column 'Point' and analyze the top 10 authors by reviews
df['Point'] = (df['Rating'] * df['No. of People rated']) / 22500
Top_authors = (df.groupby('Author')[['Point']].mean().sort_values(by='Point', ascending=False).reset_index()).head(10)
color = colours()
color = color[:9]
bar_chart(Top_authors['Author'], Top_authors['Point'], color, 'Authors', 
          'Points', 45, 'Top 10 Authors by reviews', 
          'Top 10 Authors by reviews barchart')

# Generate a scatter plot for 'Price' vs 'Rating Points'
scatter_plot(df, 'Scatter plot price vs rating')





