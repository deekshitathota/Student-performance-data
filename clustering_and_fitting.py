"""
Clustering and Fitting Assignment.

This script performs a comprehensive analysis of student performance data,
including statistical moments, data visualization, K-Means clustering, 
and linear polynomial fitting.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def plot_relational_plot(df):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Absences', y='GPA',
                    hue='GradeClass', palette='viridis', alpha=0.7)
    plt.title('Impact of Absences on Student GPA')
    plt.xlabel('Number of Absences')
    plt.ylabel('GPA')
    plt.grid(True)
    plt.savefig('relational_plot.png')
    plt.show()


def plot_categorical_plot(df):
    plt.figure(figsize=(8, 6))
    sns.barplot(data=df, x='GradeClass', y='GPA',
                estimator=np.mean, palette='coolwarm')
    plt.title('Average GPA per Grade Category')
    plt.xlabel('Grade Class')
    plt.ylabel('Mean GPA')
    plt.savefig('categorical_plot.png')
    plt.show()


def plot_statistical_plot(df):
    plt.figure(figsize=(10, 8))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.savefig('statistical_plot.png')
    plt.show()


def statistical_analysis(df, col):
    # ✅ FIXED LINE (this was your error before)
    mean = df[col].mean()

    stddev = df[col].std()
    skew = ss.skew(df[col])
    kurtosis = ss.kurtosis(df[col])

    return mean, stddev, skew, kurtosis


def preprocessing(df):
    print(df.describe())

    if 'StudentID' in df.columns:
        df = df.drop('StudentID', axis=1)

    df = df.dropna()
    return df


def writing(moments, col):
    print(f"\nAnalysis for {col}:")
    print(f"Mean = {moments[0]:.2f}")
    print(f"Std Dev = {moments[1]:.2f}")
    print(f"Skewness = {moments[2]:.2f}")
    print(f"Kurtosis = {moments[3]:.2f}")


def perform_clustering(df, col1, col2):
    data = df[[col1, col2]].values

    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)

    # Elbow plot
    inertia = []
    for k in range(1, 11):
        model = KMeans(n_clusters=k, n_init=10, random_state=42)
        model.fit(scaled)
        inertia.append(model.inertia_)

    plt.plot(range(1, 11), inertia, 'o-')
    plt.title('Elbow Method')
    plt.xlabel('k')
    plt.ylabel('Inertia')
    plt.savefig('elbow_plot.png')
    plt.show()

    # Final clustering
    model = KMeans(n_clusters=4, n_init=10, random_state=42)
    labels = model.fit_predict(scaled)

    centers = scaler.inverse_transform(model.cluster_centers_)

    return labels, data, centers


def plot_clustered_data(labels, data, centers):
    plt.figure(figsize=(10, 6))

    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
    plt.scatter(centers[:, 0], centers[:, 1],
                c='red', marker='X', s=200)

    plt.xlabel('Study Time Weekly')
    plt.ylabel('GPA')
    plt.title('Clustering')
    plt.savefig('clustering.png')
    plt.show()


def perform_fitting(df, col1, col2):
    x = df[col1].values
    y = df[col2].values

    coeffs = np.polyfit(x, y, 1)
    poly = np.poly1d(coeffs)

    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = poly(x_line)

    return x, y, x_line, y_line


def plot_fitted_data(x, y, x_line, y_line):
    plt.figure(figsize=(10, 6))

    plt.scatter(x, y, alpha=0.3)
    plt.plot(x_line, y_line, color='red')

    plt.xlabel('Absences')
    plt.ylabel('GPA')
    plt.title('Fitting Line')
    plt.savefig('fitting.png')
    plt.show()


def main():
    try:
        df = pd.read_csv('Student_performance_data _.csv')
    except:
        print("File not found")
        return

    df = preprocessing(df)

    plot_relational_plot(df)
    plot_statistical_plot(df)
    plot_categorical_plot(df)

    moments = statistical_analysis(df, 'GPA')
    writing(moments, 'GPA')

    labels, data, centers = perform_clustering(df, 'StudyTimeWeekly', 'GPA')
    plot_clustered_data(labels, data, centers)

    x, y, x_line, y_line = perform_fitting(df, 'Absences', 'GPA')
    plot_fitted_data(x, y, x_line, y_line)

    print("Done")


if __name__ == "__main__":
    main()
