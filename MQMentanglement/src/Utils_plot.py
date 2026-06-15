import matplotlib.pyplot as plt
from MQMentanglement.src.EEconstructor import entropy_as_function_of_left_endpoint, entropy_as_function_of_right_endpoint, entropy_as_function_of_center_point


def plot_entropy_left(
    phi_right: float = 5,
    phi_left_min: float = -15,
    phi_left_max: float = 4.95,
    mu: float = 0.1,
    phi_min: float = -100,
    phi_max: float = 10,
    a: float = 0.1,
    b: float = 1.0,
):
    phi_left_values, S_values = entropy_as_function_of_left_endpoint(
        phi_left_min = phi_left_min,
        phi_left_max = phi_left_max,
        mu = mu,
        phi_min = phi_min,
        phi_max = phi_max,
        a = a,
        b = b,
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(phi_left_values, S_values)

    ax.set_xlabel(r"$\phi_{\mathrm{left}}$")
    ax.set_ylabel(r"$S_{\mathrm{VN}}$")
    ax.set_title(
        rf"Liouville minisuperspace EE, $\mu={mu}$, $a={a}$, $\phi_R={phi_right}$"
    )

    fig.tight_layout()
    return fig, ax

def plot_entropy_right(
    phi_left: float = -30,
    phi_right_min: float = -14.95,
    phi_right_max: float = 5,
    mu: float = 0.1,
    phi_min: float = -100,
    phi_max: float = 10,
    a: float = 0.1,
    b: float = 1.0,
):
    phi_right_values, S_values = entropy_as_function_of_right_endpoint(
        phi_left=phi_left,
        phi_right_min=phi_right_min,
        phi_right_max=phi_right_max,
        mu=mu,
        phi_min=phi_min,
        phi_max=phi_max,
        a=a,
        b=b,
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(phi_right_values, S_values)

    ax.set_xlabel(r"$\phi_{\mathrm{right}}$")
    ax.set_ylabel(r"$S_{\mathrm{VN}}$")
    ax.set_title(
        rf"Liouville minisuperspace EE, $\mu={mu}$, $a={a}$, $\phi_L={phi_left}$"
    )

    fig.tight_layout()
    return fig, ax

def plot_entropy_center(
    phi_center_min: float = -15,
    phi_center_max: float = 5,
    Length: float = 0.5,
    mu: float = 0.1,
    phi_min: float = -70,
    phi_max: float = 20,
    a: float = 0.1,
    b: float = 1.0,
):
    phi_center_values, S_values = entropy_as_function_of_center_point(
        phi_center_min = phi_center_min,
        phi_center_max = phi_center_max,
        Length = Length,
        mu = mu,
        phi_min = phi_min,
        phi_max = phi_max,
        a = a,
        b = b,
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(phi_center_values, S_values)

    ax.set_xlabel(r"$\phi_{\mathrm{center}}$")
    ax.set_ylabel(r"$S_{\mathrm{VN}}$")
    ax.set_title(
        rf"Liouville minisuperspace EE, $\mu={mu}$, $a={a}$, $L={Length}$"
    )

    fig.tight_layout()
    return fig, ax


if __name__ == "__main__":
    fig, ax = plot_entropy_left()
    plt.show()