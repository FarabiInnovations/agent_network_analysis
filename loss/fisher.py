# ------- #
# Fisher Information
# Negative expected value of the second derivitave of the log-likelihood
# Gives us an indication, the higher the Fisher information the more confident we might be about 
# the reliability of the parameter.   
# In tha case of transformer output to the business, we are considering useful/not useful output. 
# In which case we are modelling this a Bernoulli distribution, defining the variance p(1-p)
# and Fisher information 1/p(1-p) of the system and each node. This will indicate
# how reliable each node (or the complete system) is in both its reliability to succeed or fail.
# 
# ------- #

def bernoulli_stats(success_count, n):
    """
    Computes the estimated probability, variance, and Fisher information
    for a Bernoulli process.

    Parameters:
        success_count (int or float): The count of useful outputs (successes)
        n (int): Total number of trials.

    Returns:
        tuple: (p_bar, variance, fisher_info)
            - p_bar (float): Estimated probability of success (success_count/n)
            - variance (float): Variance, computed as p_bar * (1 - p_bar)
            - fisher_info (float): Fisher information, computed as 1 / (p_bar*(1 - p_bar))
              (returns float('inf') if variance is 0 to avoid division by zero)
    """
    if n <= 0:
        raise ValueError("n trials must be greater than 0")
    
    p_bar = success_count / n
    variance = p_bar * (1 - p_bar)
    fisher_info = float('inf') if variance == 0 else 1 / variance

    return p_bar, variance, fisher_info


# def gaussian_stats(success_count, n):
# def poisson_stats(success_count, n):
# def geometric_stats(success_count, n):


# if __name__ == '__main__':
#     success_count = 80
#     total_trials = 100
#     p_bar, variance, fisher_info = bernoulli_stats(success_count, total_trials)
#     print(f"p_bar: {p_bar:.4f}")
#     print(f"Variance: {variance:.4f}")
#     print(f"Fisher Information: {fisher_info:.4f}")
