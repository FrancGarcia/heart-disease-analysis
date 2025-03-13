import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, f_oneway, pearsonr


def compute_plot_heart_disease_associations(df,num_vars,cat_vars):
    '''
    Check the relationship of Heart Disease Status(categorical) with other factors
    '''


    results = []

    if df['Heart Disease Status'].dtype == 'O':
        df['Heart Disease Status'] = df['Heart Disease Status'].map({'Yes': 1, 'No': 0})

    # Perform one-way ANOVA for numerical variables.
    for var in num_vars:
        group_w_hd = df[df["Heart Disease Status"] == 1][var]
        group_wo_hd = df[df["Heart Disease Status"] == 0][var]
        
        if len(group_w_hd) > 1 and len(group_wo_hd) > 1:
            f_stat, p_val = f_oneway(group_w_hd, group_wo_hd)
            results.append({
                "Variable": var,
                "Test": "ANOVA",
                "p-value": p_val
            })

    # Perform chi-square test for categorical variables.
    for var in cat_vars:
        # Create contingency table between the factor and Heart Disease Status.
        contingency = pd.crosstab(df[var], df["Heart Disease Status"])
        if contingency.shape[0] > 1 and contingency.shape[1] > 1:
            chi2, p_val, dof, expected = chi2_contingency(contingency)
            results.append({
                "Variable": var,
                "Test": "chi-square",
                "p-value": p_val
            })

    results_df = pd.DataFrame(results)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=results_df, x="p-value", y="Variable", size="p-value", hue="p-value",
                    sizes=(50, 500), palette="viridis", edgecolor="black", alpha=0.8)
    plt.xlabel("p-value")
    plt.title("Strength of Association with Heart Disease")
    plt.legend()
    plt.show()

    return results_df

def compute_plot_indicators_associations_heatmap(df,outcomes, num_vars,cat_vars):
    '''
    
    '''

    predictors = list(set(num_vars + cat_vars))

    for outcome in outcomes:
        if outcome in predictors:
            predictors.remove(outcome)
    
    association_results = {outcome: {} for outcome in outcomes}
    
    for outcome in outcomes:
        for factor in predictors:
            # categorical outcomes
            if outcome in ["Stress Level", "Gender"]:
                if df[factor].dtype in ['object', 'category']:
                    # Both categorical: chi-square
                    contingency = pd.crosstab(df[factor], df[outcome])
                    chi2, p_val, dof, expected = chi2_contingency(contingency)
                    association_results[outcome][factor] = p_val
                else:
                    # Numerical: one-way ANOVA across stress level groups.
                    groups = [df[df[outcome] == level][factor] for level in df[outcome].unique()]
                    f_stat, p_val = f_oneway(*groups)
                    association_results[outcome][factor] = p_val
            # numerical outcomes
            else:
                if df[factor].dtype not in ['object', 'category']:
                    # Both numerical: Pearson correlation.
                    data = df[[factor, outcome]]
                    r, p_val = pearsonr(data[factor], data[outcome])
                    association_results[outcome][factor] = p_val
                else:
                    # Categorical: one-way ANOVA comparing outcome across categories.
                    groups = [df[df[factor] == level][outcome] for level in df[factor].unique()]
                    f_stat, p_val = f_oneway(*groups)
                    association_results[outcome][factor] = p_val
    
    assoc_df = pd.DataFrame(association_results)
    
    
    plt.figure()
    sns.heatmap(assoc_df, annot=True, cmap="coolwarm")
    plt.title("Strength of Associations")
    plt.xlabel("Outcome")
    plt.ylabel("Predictor Factor")
    plt.tight_layout()
    plt.show()
    
    return assoc_df