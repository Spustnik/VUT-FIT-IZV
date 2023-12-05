from typing import Callable, List
from bs4 import BeautifulSoup
import requests
import numpy as np
import matplotlib.pyplot as plt


def integrate(f: Callable[[np.ndarray], np.ndarray], a: float, b: float, steps=1000) -> float:
    """
    Calculate integral of f from a to b using steps
    f: Function to integrate
    a: Lower bound
    b: Upper bound
    steps: Number of steps
    Return the value of the integral
    """
    x = np.linspace(a, b, steps)
    Part_01 = (x[:-1] + x[1:]) / 2  # Calculate midpoints
    Part_02 = x[1:] - x[:-1]
    return np.sum(f(Part_01) * Part_02)


def generate_graph(a: List[float], show_figure: bool = False, save_path: str | None = None):
    """
    Generate and visualize the function f_a(x) for specified 'a' values.
    a: List of 'a' values.
    show_figure: If True, the figure will be displayed.
    save_path: If specified, the figure will be saved to the specified path.
    """
    x = np.linspace(-3, 3, 1000)
    f_matrix = np.zeros((len(a), len(x)))  # Create matrix for f_a(x) values

    for i, ai in enumerate(a):
        f_matrix[i] = ai**2 * x**3 * np.sin(x)  # Calculate f_a(x) values

    plt.figure(figsize=(10, 6))
    colors = ['blue', 'orange', 'green']

    for i, ai in enumerate(a):  # Plot f_a(x) values
        f_values = ai**2 * x**3 * np.sin(x)
        f_integral = np.trapz(f_values, x)
        plt.plot(x, f_matrix[i], label=f'$Y_{{{ai:.1f}}}(x)$', color=colors[i])
        plt.fill_between(x, f_matrix[i], color=colors[i], alpha=0.1)
        text = "$\\int f_{{%.1f}}(x)dx = %.2f$" % (ai, f_integral)
        plt.annotate(text,
                     xy=(3.01,
                         np.multiply(i, np.multiply(3, 3)) + 1), size=8)

        plt.xlim(-3, 3.99)
        plt.ylim(0, 40)

    plt.xlabel('x')
    plt.ylabel(r'$f_{a}(x)$', fontsize=14)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=len(a))

    if show_figure:
        plt.show()

    if save_path:
        plt.savefig(save_path)

    integrals = np.trapz(f_matrix, x)

    return integrals


def generate_sinus(show_figure: bool = False, save_path: str | None = None):
    """
    Generate and visualize the function f1(t) and f2(t) and their sum.
    show_figure: If True, the figure will be displayed.
    save_path: If specified, the figure will be saved to the specified path.

    """

    t = np.linspace(0, 100, 100000)
    fig, axs = plt.subplots(3, 1, figsize=(8, 10))

    f1_values = 0.5 * np.cos(1 / 50 * np.pi * t)
    f2_values = 0.25 * (np.sin(np.pi * t) + np.sin(1.5 * np.pi * t))

    sum_values = f1_values + f2_values
    plt.subplots_adjust(hspace=0.5)

    # Plot f1
    axs[0].plot(t, f1_values, label='f1(t)')
    axs[0].set_xlabel('t')
    axs[0].set_ylabel(r'$f_{1}(t)$', fontsize=14)
    axs[0].margins(0, 0)
    axs[0].yaxis.set_ticks(np.arange(-0.8, 1.0, 0.4))

    # Plot f2
    axs[1].plot(t, f2_values, label='f2(t)')
    axs[1].set_xlabel('t')
    axs[1].set_ylabel(r'$f_{2}(t)$', fontsize=14)
    axs[1].margins(0, 0)
    axs[1].yaxis.set_ticks(np.arange(-0.8, 1.0, 0.4))

    mask = sum_values > f1_values

    axs[2].plot(t, sum_values, color='red', linewidth=1)
    axs[2].plot(t, np.ma.masked_less(sum_values, f1_values),
                color='green', linewidth=1)
    axs[2].set_xlabel('t')
    axs[2].set_ylabel(r'$f_{1}(t) + f_{2}(t)$', fontsize=14)
    axs[2].margins(0, 0)
    axs[2].yaxis.set_ticks(np.arange(-0.8, 1.0, 0.4))

    if save_path:
        plt.savefig(save_path)

    if show_figure:
        plt.show()

    plt.close()


def download_data() -> list[dict[str, any]]:
    """
    Scrape meteorological station data from the specified website.
    """
    url = "https://ehw.fit.vutbr.cz/izv/st_zemepis_cz.html"
    r = requests.get(url)
    r.raise_for_status()
    content = BeautifulSoup(r.content, "html.parser")
    rows = content.find_all("tr", class_="nezvyraznit")
    data = []
    for row in rows:
        columns = row.find_all("td")
        position = columns[0].strong.text.strip()
        lat = float(columns[2].text.replace(',', '.').strip()[
                    :-1])
        long = float(columns[4].text.replace(',', '.').strip()[
                     :-1])
        height = float(columns[6].text.replace(',', '.').strip())

        entry = {
            "position": position,
            "lat": lat,
            "long": long,
            "height": height
        }

        data.append(entry)
    return data


if __name__ == '__main__':
    generate_graph([1.0, 1.5, 2.0], show_figure=True)
