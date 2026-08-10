import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

from load.data_ingest import IngestData
from src.check import InspectData

if __name__ == "__main__":

    data_load=IngestData("data/ab_testing_data.csv")
    data=data_load.open_csv()
    #data_inspect=InspectData(data)
    #head=data_inspect.check_data()
    #info=data_inspect.check_info()
    #change_columns_name=data_inspect.change_col()
    #infon=data_inspect.check_info()
    
    alpha = 0.05
    delta = 0.1
    #Total number of user per groups  
    N_con = len(data[data["group"] == "con"])
    N_exp = len(data[data["group"] == "exp"])
    #N_con = (data["group"] == "con").sum()
    #N_exp = (data["group"] == "exp").sum()
    
    #Caculation the total number of click per groups  
    X_con = data.groupby("group")["click"].sum().loc["con"]
    X_exp = data.groupby("group")["click"].sum().loc["exp"]

    #Caculation conversion rate of groups
    p_con = X_con/N_con
    p_exp = X_exp/N_exp

    #Caculation pooled proportion
    p_pool = (X_con + X_exp) / (N_con + N_exp)
              
    #Caculation pool varience
    pool_var = p_pool*(1-p_pool)*(1/N_con + 1/N_exp)
    
    #Computing the standaed error
    SE = np.sqrt(pool_var)

    #Caculate z score 
    z = (p_exp - p_con)/SE

    #Critical value 
    z_crit = norm.ppf(1-alpha/2)
    
    #Caculate p value (For a two-tailed test, 2 *)
    p_value = 2 * norm.sf(abs(z))

    def compare(p_value, alpha):
        if p_value <= alpha:
            print("Statistically significant: Reject H0")
        else:
            print("Not statistically significant: Do not reject H0")
    compare(p_value, alpha)

    #Parameters for standard normal distribution 
    mu = 0 #mean
    sigma = 1 #std 
    x = np.linspace(mu - 3*sigma,mu + 3*sigma,100)
    #norm.pdf() calculates the probability density.
    y = norm.pdf(x, mu, sigma)

    #Visualization
    #Draw the normal distribution
    plt.plot(x, y, label='Standard Normal Distribution')
    #Highlight the rejection regions
    plt.fill_between(
            x,
            y,
            where=(x > z_crit) | (x < -z_crit),
            color='red',
            alpha=0.5,
            label='Rejection Region'
                    )
    #Show the test statistic
    plt.axvline(
            z,
            color='green',
            linestyle='dashed',
            linewidth=2,
            label=f'Test Statistic = {z:.2f}'
                    )
    #Show the critical values
    plt.axvline(z_crit,color='green',linestyle='dashed',linewidth=2)
    plt.axvline(-z_crit,color='green',linestyle='dashed',linewidth=2)

    plt.xlabel('Z-value')
    plt.ylabel('Probability Density')
    plt.title('Gaussian Distribution with Rejection Region')
    plt.legend()
    plt.show()

    CI = [
            round((p_exp - p_con) - SE * z_crit, 3),
            round((p_exp - p_con) + SE * z_crit, 3)
                    ]
    print("Confidence Interval of the 2 sample Z-test is: ", CI)

    def is_Practically_significant(delta, CI_95):
        lower_bound_CI = CI_95[0]
        if lower_bound_CI >= delta:
            print(f"We have practical significance.")
            return True
        else:
            print("We don't have practical significance")
            return False
    
    is_Practically_significant(delta, CI)
    
                
    

    



