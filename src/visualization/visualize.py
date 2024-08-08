import matplotlib.pyplot as plt
import seaborn as sns


def plot_scatterplot(data, x, y, title=None, hue=None):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data, x=x, y=y, hue=hue)
    plt.xticks(rotation=90)
    if title:
        plt.title(title)
    plt.show()


def plot_countplot(data, x, title=None):
    plt.figure(figsize=(10, 6))
    sns.countplot(data=data, x=x)
    plt.xticks(rotation=90)
    if title:
        plt.title(title)
    plt.show()


def plot_barplot(data, x, y, title=None):
    plt.figure(figsize=(10, 6))
    sns.barplot(data=data, x=x, y=y)
    plt.xticks(rotation=90)
    if title:
        plt.title(title)
    plt.show()


def plot_regplot(data, x, y, title=None):
    plt.figure(figsize=(10, 6))
    sns.regplot(data=data, x=x, y=y)
    if title:
        plt.title(title)
    plt.show()


def plot_violinplot(data, x, y, title=None):
    plt.figure(figsize=(12, 6))
    sns.violinplot(data=data, x=x, y=y)
    plt.xticks(rotation=90)
    if title:
        plt.title(title)
    plt.show()


def plot_facetgrid(data, x, y, col, row, title=None):
    plt.figure(figsize=(12, 8))
    grid = sns.FacetGrid(data, col=col, row=row)
    grid = grid.map(plt.scatter, x, y)
    if title:
        plt.suptitle(title)
    plt.show()


def plot_piechart(data, column, title=None):
    counts = data[column].value_counts()
    plt.figure(figsize=(8, 6))
    plt.pie(counts, labels=counts.index, autopct="%1.1f%%")
    plt.axis("equal")
    if title:
        plt.title(title)
    plt.show()


def plot_kdeplot(data, x, y, title=None):
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=data, x=x, y=y, cmap="Reds", fill=True)
    if title:
        plt.title(title)
    plt.show()


def plot_lineplot(data, x, y, title=None):
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=data, x=x, y=y)
    plt.xticks(rotation=90)
    if title:
        plt.title(title)
    plt.show()


def plot_heatmap(data, title=None):
    plt.figure(figsize=(10, 6))
    sns.heatmap(data)
    if title:
        plt.title(title)
    plt.show()


def plot_boxplot(data, x, y, title=None):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data, x=x, y=y)
    if title:
        plt.title(title)
    plt.show()
