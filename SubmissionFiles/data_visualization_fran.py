import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
def produce_pie_charts(dataframe: pd.DataFrame, independent_feature: str, dependent_feature: str):
    """
    Produces pie chart of percentage of people with
    the feature indicators that have heart disease.

    :param dataframe: The dataframe to produce the data from.
    :param dependent_feature: The column that depends on the other to create pie chart.
    :param independent_feature: The column that influences the other to create the pie chart.

    :return: Produce a pie chart between the independent and dependent features.
    """
    assert(isinstance(dataframe, pd.DataFrame) and dataframe.shape[0] > 0), "First argument must be a valid DataFrame with at least 1 row"
    assert(isinstance(independent_feature, str) and independent_feature in dataframe.columns), "The second argument must be a valid string feature in the dataframe"
    assert(isinstance(dependent_feature, str) and dependent_feature in dataframe.columns), "The third argument must be a valid string feature in the dataframe"
    df_heart_disease = dataframe[dataframe[dependent_feature].str.lower() == 'yes']
    print(df_heart_disease.shape)
    ind_counts = df_heart_disease[independent_feature].value_counts()
    print(ind_counts)

    plt.figure(figsize=(6,6))
    plt.pie(ind_counts, labels=ind_counts.index, autopct='%1.1f%%')
    plt.title(f"{independent_feature} distribution in People with Heart Disease")

def create_histoplot(dataframe: pd.DataFrame, x: str, hue: str, cat: str = None):
    """
    Creates a histoplot given the dataframe, col, x and hue.

    :param dataframe: The dataframe to extract and visualize
    :param: The column to place into the FacetGrid
    :param x: The x value to place into the histoplot
    :param hue: The hue to place into the FacetGrid
    """
    assert(isinstance(dataframe, pd.DataFrame) and dataframe.shape[0] > 0), "First argument must be a valid dataframe"
    assert(isinstance(x, str) and len(x) > 0 and x in dataframe.columns), "Second argument must be valid feature in columns of dataframe"
    assert(isinstance(hue, str) and len(hue) > 0 and hue in dataframe.columns), "Third argument must be valid feature in columns of dataframe"
    assert(isinstance(cat, str) and len(cat) > 0 and cat in dataframe.columns), "Fourth argument must be valid feature in columns of dataframe"

    if cat == "Cholesterol Category":
        # 3-variable distribution with Cholesterol 
        g = sns.FacetGrid(dataframe, col=cat, hue=hue, height=4, aspect=1.2, sharex=True, sharey=True, palette={"No": "darkred", "Yes": "darkgreen"}, hue_order=["Yes", "No"], col_order=dataframe[col].unique())
        g.map(sns.histplot, x, kde=True, bins=30, stat="density", common_norm=False)
        g.add_legend()
        if cat:
            for ax in g.axes.flat:
                if ax.get_title() == "Cholesterol Category = Normal":
                    ax.text(0.5, 0.97, "Normal: < 200", ha="center", va="center", transform=ax.transAxes, fontsize=10, color="black")
                elif ax.get_title() == "Cholesterol Category = Elevated":
                    ax.text(0.5, 0.97, "Elevated: 200 - 239", ha="center", va="center", transform=ax.transAxes, fontsize=10, color="black")
                elif ax.get_title() == "Cholesterol Category = High":
                    ax.text(0.5, 0.97, "High: > 240", ha="center", va="center", transform=ax.transAxes, fontsize=10, color="black")
        plt.show()
        return
    elif cat != "Cholesterol Category" and cat is not None:
        # 3-variable distribution that is NOT Cholesterol
        g = sns.FacetGrid(dataframe, col=cat, hue=hue, height=4, aspect=1.2, sharex=True, sharey=True, palette={"No": "darkred", "Yes": "darkgreen"}, hue_order=["Yes", "No"], col_order=dataframe[col].unique())
        g.map(sns.histplot, x, kde=True, bins=30, stat="density", common_norm=False)
        g.add_legend()
        plt.show()
        return
    elif cat is None:
        # 2-variable distribution
        g = sns.histplot(dataframe, x=x, hue=hue, kde=True, bins=30, stat="density", common_norm=False, hue_order=["Yes", "No"], palette={"Yes": "darkgreen", "No": "darkred"})
        plt.title(f"{x} Distribution by {hue} Distribution")
        plt.show()
        return

if __name__ == "__main__":
    df = pd.read_csv("../data/equal_distribution_hds.csv")

    create_histoplot(df, "BMI", "Heart Disease Status")
    create_histoplot(df, "BMI", "Heart Disease Status", col="Cholesterol Category")

    create_histoplot(df, "Stress Level", "Heart Disease Status")
    create_histoplot(df, "Stress Level", "Heart Disease Status", col="Alcohol Consumption")

    create_histoplot(df, "Blood Pressure", "Heart Disease Status")
    create_histoplot(df, "Blood Pressure Category", "Heart Disease Status", col="Diabetes")

    produce_pie_charts(df, "Stress Level", "Heart Disease Status")
    produce_pie_charts(df, "High Blood Pressure", "Heart Disease Status")
    produce_pie_charts(df, 'Low HDL Cholesterol', 'Heart Disease Status')
    produce_pie_charts(df, "High LDL Cholesterol", "Heart Disease Status")