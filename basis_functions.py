import torch
import math


class BasisFunctions(object):
    def __len__(self):
        """Number of basis functions."""
        raise NotImplementedError

    def evaluate(self, t):
        raise NotImplementedError

    def integrate_t2_times_psi(self, a, b):
        """Compute integral int_a^b (t**2) * psi(t)."""
        raise NotImplementedError

    def integrate_t_times_psi(self, a, b):
        """Compute integral int_a^b t * psi(t)."""
        raise NotImplementedError

    def integrate_psi(self, a, b):
        """Compute integral int_a^b psi(t)."""
        raise NotImplementedError

class GaussianBasisFunctions(BasisFunctions):
    """Function phi(t) = Gaussian(t; mu, sigma_sq)."""

    def __init__(self, mu, sigma):
        self.mu = mu.unsqueeze(0)
        self.sigma = sigma.unsqueeze(0)

    def __repr__(self):
        return f"GaussianBasisFunction(mu={self.mu}, sigma={self.sigma})"

    def __len__(self):
        """Number of basis functions."""
        return self.mu.size(1)

    def _phi(self, t):
        return 1.0 / math.sqrt(2 * math.pi) * torch.exp(-0.5 * t ** 2)

    def _Phi(self, t):
        return 0.5 * (1 + torch.erf(t / math.sqrt(2)))

    def _integrate_product_of_gaussians(self, mu, sigma_sq):
        sigma = torch.sqrt(self.sigma ** 2 + sigma_sq)
        return self._phi((mu - self.mu) / sigma) / sigma

    def evaluate(self, t):
        return self._phi((t - self.mu) / self.sigma) / self.sigma

    def integrate_t2_times_psi(self, a, b):
        """Compute integral int_a^b (t**2) * psi(t)."""
        return (
            (self.mu ** 2 + self.sigma ** 2)
            * (
                self._Phi((b - self.mu) / self.sigma)
                - self._Phi((a - self.mu) / self.sigma)
            )
            - (
                self.sigma
                * (b + self.mu)
                * self._phi((b - self.mu) / self.sigma)
            )
            + (
                self.sigma
                * (a + self.mu)
                * self._phi((a - self.mu) / self.sigma)
            )
        )

    def integrate_t_times_psi(self, a, b):
        """Compute integral int_a^b t * psi(t)."""
        return self.mu * (
            self._Phi((b - self.mu) / self.sigma)
            - self._Phi((a - self.mu) / self.sigma)
        ) - self.sigma * (
            self._phi((b - self.mu) / self.sigma)
            - self._phi((a - self.mu) / self.sigma)
        )

    def integrate_psi(self, a, b):
        """Compute integral int_a^b psi(t)."""
        return self._Phi((b - self.mu) / self.sigma) - self._Phi(
            (a - self.mu) / self.sigma
        )

    def integrate_t2_times_psi_gaussian(self, mu, sigma_sq):
        """Compute integral int N(t; mu, sigma_sq) * t**2 * psi(t)."""
        S_tilde = self._integrate_product_of_gaussians(mu, sigma_sq)
        mu_tilde = (self.mu * sigma_sq + mu * self.sigma ** 2) / (
            self.sigma ** 2 + sigma_sq
        )
        sigma_sq_tilde = ((self.sigma ** 2) * sigma_sq) / (
            self.sigma ** 2 + sigma_sq
        )
        return S_tilde * (mu_tilde ** 2 + sigma_sq_tilde)

    def integrate_t_times_psi_gaussian(self, mu, sigma_sq):
        """Compute integral int N(t; mu, sigma_sq) * t * psi(t)."""
        S_tilde = self._integrate_product_of_gaussians(mu, sigma_sq)
        mu_tilde = (self.mu * sigma_sq + mu * self.sigma ** 2) / (
            self.sigma ** 2 + sigma_sq
        )
        return S_tilde * mu_tilde

    def integrate_psi_gaussian(self, mu, sigma_sq):
        """Compute integral int N(t; mu, sigma_sq) * psi(t)."""
        return self._integrate_product_of_gaussians(mu, sigma_sq)