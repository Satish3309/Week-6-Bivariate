import pandas as pd
import numpy as np
class Univariate():
    def quan_qual(dataset):
        quan=[]
        qual=[]
        for column_names in dataset.columns:
            if dataset[column_names].dtypes=="object":
                qual.append(column_names)

            else:
                quan.append(column_names)

        return quan,qual
    
    def freq_Tab(dataset,column_names):
        frequency_Tab=pd.DataFrame(columns=["Unique_values","Frequency","Relative_Frequency","cumsum"])
        frequency_Tab["Unique_values"]=dataset[column_names].value_counts().index
        frequency_Tab["Frequency"]=dataset[column_names].value_counts().values
        frequency_Tab["Relative_Frequency"]=frequency_Tab["Frequency"]/frequency_Tab["Frequency"].sum()
        frequency_Tab["cumsum"]=frequency_Tab["Relative_Frequency"].cumsum()
        return frequency_Tab
    
    def detect_lesser_greater(quan,descriptive):
        lesser_de=[]
        greater_de=[]
        for column_names in quan:
            if(descriptive[column_names]["Mini"]<descriptive[column_names]["lesser_per"]):
                lesser_de.append(column_names)
            #here we don't use else or elif because if the above condition is true else or elife conditions won't work
            if(descriptive[column_names]["Max"]>descriptive[column_names]["greater_per"]):
                greater_de.append(column_names)
        return lesser_de,greater_de
    
    def descriptive_Tab(dataset,quan):
        descriptive=pd.DataFrame(index=["Mean","Median","Mode","Q1:25%","Q2:50%","Q3:75%","99%","Q4:100%","IQR","1.5rule",
                                    "lesser_per","greater_per","Mini","Max"],columns=quan)  
        for column_names in quan:                           
            descriptive[column_names]["Mean"]=dataset[column_names].mean()
            descriptive[column_names]["Median"]=dataset[column_names].median()
            descriptive[column_names]["Mode"]=dataset[column_names].mode()[0]
            #Before start we should create columns and rows to save value right?
            descriptive[column_names]["Q1:25%"]=dataset.describe()[column_names]["25%"]
            descriptive[column_names]["Q2:50%"]=dataset.describe()[column_names]["50%"]
            descriptive[column_names]["Q3:75%"]=dataset.describe()[column_names]["75%"]
            descriptive[column_names]["99%"]   =np.percentile(dataset[column_names],99)
            descriptive[column_names]["Q4:100%"]=dataset.describe()[column_names]["max"]
            #so,now we are going to add IQR values to fing outliers. Add column and row to the dataframe.
            descriptive[column_names]["IQR"]=descriptive[column_names]["Q3:75%"]-descriptive[column_names]["Q1:25%"]
            descriptive[column_names]["1.5rule"]=1.5*descriptive[column_names]["IQR"]
            descriptive[column_names]["lesser_per"]=descriptive[column_names]["Q1:25%"]-descriptive[column_names]["1.5rule"]
            descriptive[column_names]["greater_per"]=descriptive[column_names]["Q3:75%"]+descriptive[column_names]["1.5rule"]
            descriptive[column_names]["Mini"]=dataset[column_names].min()
            descriptive[column_names]["Max"]=dataset[column_names].max()
            descriptive[column_names]["skew"]=dataset[column_names].skew()
            descriptive[column_names]["kurtosis"]=dataset[column_names].kurtosis()
        return descriptive #while convert logic into function position of return statement is important 
    
    def Replace_outliers(dataset,lesser_de,greater_de,descriptive):
        for column_names in lesser_de:
            dataset[column_names][dataset[column_names]<descriptive[column_names]["lesser_per"]]=descriptive[column_names]
            ["lesser_per"]
        for column_names in greater_de:
            dataset[column_names][dataset[column_names]>descriptive[column_names]["greater_per"]]=descriptive[column_names]
            ["greater_per"]
        return dataset
    
    def get_pdf_probability(dataset,startrange,endrange): #(make function for future use)
        from matplotlib import pyplot   #(import matplotlib library for handling diagram)
        from scipy.stats import norm    #(import norm to plaot narmal distribution curve)
        import seaborn as sns
        #(kernel density=true,this function from seaborn used to plot curve)
        ax = sns.distplot(dataset,kde=True,kde_kws={'color':'blue'},color='Green') 
        #then, the following code created for ploting vertical line with red colour
        pyplot.axvline(startrange,color='Red')
        pyplot.axvline(endrange,color='Red')
        # generate a sample
        sample = dataset #(assing dataset to sample variable)
        # calculate parameters
        sample_mean =sample.mean()
        sample_std = sample.std()
        print('Mean=%.3f, Standard Deviation=%.3f' % (sample_mean, sample_std))
        # define the distribution
        dist = norm(sample_mean, sample_std) 
        # sample probabilities for a range of outcomes
        values = [value for value in range(startrange, endrange)]
        probabilities = [dist.pdf(value) for value in values]    
        prob=sum(probabilities)
        print("The area between range({},{}):{}".format(startrange,endrange,sum(probabilities)))
        return prob
    
    
    def std_nb_plot(dataset): # Coverted to standard Normal Distribution
    
        import seaborn as sns
        mean=dataset.mean()
        std=dataset.std()

        values=[i for i in dataset]   #we pass dataset to convert as list so,we can take any column for output

        z_score=[((j-mean)/std) for j in values]  #we know formulae z_score= (x-mean / std)

        sns.distplot(z_score,kde=True) #now plot z_score distribution 

        sum(z_score)/len(z_score)
        return z_score

    