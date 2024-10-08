import pandas as pd  # data manipulation and analysis
import seaborn as sns

sns.set_theme()


def plot_consumption(
    data: pd.DataFrame, year: int, col_wrap: int = 4, title_suffix: str = None
):
    def month_titles_for_year(year):
        return {
            1: f"Jan {year}",
            2: f"Feb {year}",
            3: f"Mar {year}",
            4: f"Apr {year}",
            5: f"May {year}",
            6: f"Jun {year}",
            7: f"Jul {year}",
            8: f"Aug {year}",
            9: f"Sep {year}",
            10: f"Oct {year}",
            11: f"Nov {year}",
            12: f"Dec {year}",
        }

    month_titles = month_titles_for_year(year)

    g = sns.relplot(
        data=data,
        kind="line",
        x="time_of_day",
        y="total_home_usage",
        hue="day_of_week",
        col="month",
        col_wrap=col_wrap,
        facet_kws={"palette": ["red"]},
    )

    for ax in g.axes.flat:
        ax.tick_params(
            axis="x", labelbottom=True
        )  # Thi show x-axis labels on the bottom for all subplots
        ax.set_ylabel(
            "Consumption (kWh)", labelpad=10, fontsize=14
        )  # set y-axis label font size
        ax.set_xlabel(
            "Time of Day", labelpad=15, fontsize=14
        )  # set x-axis label font size

        # Set x-axis tick label font size and rotation
        ax.tick_params(axis="x", labelsize=12)

        # Set x-axis and x-axis limits
        ax.set_xlim([0, 86400])  # set x-axis limits to 0-24
        ax.set_ylim(0, 1000)

        # Set custom x-axis tick labels
        ax.set_xticks([21600, 43200, 64800])  # set custom tick locations
        ax.set_xticklabels(["6am", "Noon", "6pm"])  # set custom tick labels

        # Get the current month title
        month = int(ax.get_title().split("=")[1])

        # Customize the month title using the dictionary
        ax.set_title(
            month_titles[month] + (f" ({title_suffix})" if title_suffix else ""),
            pad=20,
            fontsize=16,
        )

    g.tight_layout()
    return g
