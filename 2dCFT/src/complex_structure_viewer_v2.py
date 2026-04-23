import numpy as np
import matplotlib.pyplot as plt
from mpmath import mp

mp.dps = 20


class ComplexStructureViewer:
    """
    Heuristic visualizer for the analytic structure of a complex function.
    """

    def __init__(self, func, xlim=(-2, 2), ylim=(-2, 2), resolution=220):
        self.func = func
        self.xlim = xlim
        self.ylim = ylim
        self.resolution = int(resolution)
        self._sample_cache = None
        self._diag_cache = None

    @staticmethod
    def _safe_mpc(x, y):
        # mpmath can be slow here, but this is still the safest generic path
        # for special functions that expect mp numbers.
        return mp.mpc(float(x), float(y))

    @staticmethod
    def wrapped_angle_diff(a, b):
        return np.angle(np.exp(1j * (a - b)))

    def clear_cache(self):
        self._sample_cache = None
        self._diag_cache = None

    def sample(self, force=False):
        if (self._sample_cache is not None) and (not force):
            return self._sample_cache

        x = np.linspace(self.xlim[0], self.xlim[1], self.resolution)
        y = np.linspace(self.ylim[0], self.ylim[1], self.resolution)
        X, Y = np.meshgrid(x, y)

        F = np.full(X.shape, np.nan + 1j * np.nan, dtype=np.complex128)
        finite_mask = np.zeros(X.shape, dtype=bool)

        for i in range(self.resolution):
            for j in range(self.resolution):
                z = self._safe_mpc(X[i, j], Y[i, j])
                try:
                    val = self.func(z)
                    val_complex = complex(val)
                    if np.isfinite(val_complex.real) and np.isfinite(val_complex.imag):
                        F[i, j] = val_complex
                        finite_mask[i, j] = True
                except Exception:
                    pass

        self._sample_cache = (X, Y, F, finite_mask)
        return self._sample_cache

    @staticmethod
    def _local_extremum_masks(logabs):
        """
        Very small 3x3 local test:
          - likely zero if center is distinctly smaller than neighbors
          - likely pole if center is distinctly larger than neighbors
        """
        n0, n1 = logabs.shape
        zero_like = np.zeros_like(logabs, dtype=bool)
        pole_like = np.zeros_like(logabs, dtype=bool)

        for i in range(1, n0 - 1):
            for j in range(1, n1 - 1):
                c = logabs[i, j]
                if not np.isfinite(c):
                    continue
                hood = logabs[i - 1:i + 2, j - 1:j + 2]
                vals = hood[np.isfinite(hood)]
                if vals.size < 5:
                    continue
                med = np.median(vals)
                if c < med - 0.35:
                    zero_like[i, j] = True
                if c > med + 0.35:
                    pole_like[i, j] = True

        return zero_like, pole_like

    def compute_diagnostics(self, force=False):
        if (self._diag_cache is not None) and (not force):
            return self._diag_cache

        X, Y, F, finite_mask = self.sample(force=force)

        absF = np.abs(F)
        argF = np.angle(F)

        with np.errstate(divide='ignore', invalid='ignore'):
            logabs = np.log10(absF)

        winding = np.full((self.resolution - 1, self.resolution - 1), np.nan)
        cell_valid = np.zeros_like(winding, dtype=bool)

        for i in range(self.resolution - 1):
            for j in range(self.resolution - 1):
                vals = [
                    argF[i, j],
                    argF[i, j + 1],
                    argF[i + 1, j + 1],
                    argF[i + 1, j],
                ]
                if np.all(np.isfinite(vals)):
                    d1 = self.wrapped_angle_diff(vals[1], vals[0])
                    d2 = self.wrapped_angle_diff(vals[2], vals[1])
                    d3 = self.wrapped_angle_diff(vals[3], vals[2])
                    d4 = self.wrapped_angle_diff(vals[0], vals[3])
                    total = d1 + d2 + d3 + d4
                    winding[i, j] = total / (2 * np.pi)
                    cell_valid[i, j] = True

        Xc = 0.25 * (X[:-1, :-1] + X[:-1, 1:] + X[1:, :-1] + X[1:, 1:])
        Yc = 0.25 * (Y[:-1, :-1] + Y[:-1, 1:] + Y[1:, :-1] + Y[1:, 1:])

        # Meromorphic winding guess
        zero_mask = cell_valid & (np.abs(winding - 1) < 0.35)
        pole_mask = cell_valid & (np.abs(winding + 1) < 0.35)

        # Suppress fake detections near branch points by also asking for local
        # modulus behavior in the corresponding cell center.
        zero_like_px, pole_like_px = self._local_extremum_masks(logabs)
        zero_like_cell = zero_like_px[:-1, :-1] | zero_like_px[:-1, 1:] | zero_like_px[1:, :-1] | zero_like_px[1:, 1:]
        pole_like_cell = pole_like_px[:-1, :-1] | pole_like_px[:-1, 1:] | pole_like_px[1:, :-1] | pole_like_px[1:, 1:]

        zero_mask &= zero_like_cell
        pole_mask &= pole_like_cell

        phase_jump = np.zeros_like(argF, dtype=float)
        count = np.zeros_like(argF, dtype=float)

        horiz = np.abs(self.wrapped_angle_diff(argF[:, 1:], argF[:, :-1]))
        vert = np.abs(self.wrapped_angle_diff(argF[1:, :], argF[:-1, :]))

        phase_jump[:, 1:] += np.where(np.isfinite(horiz), horiz, 0.0)
        phase_jump[:, :-1] += np.where(np.isfinite(horiz), horiz, 0.0)
        count[:, 1:] += np.isfinite(horiz)
        count[:, :-1] += np.isfinite(horiz)

        phase_jump[1:, :] += np.where(np.isfinite(vert), vert, 0.0)
        phase_jump[:-1, :] += np.where(np.isfinite(vert), vert, 0.0)
        count[1:, :] += np.isfinite(vert)
        count[:-1, :] += np.isfinite(vert)

        with np.errstate(invalid='ignore', divide='ignore'):
            phase_jump_score = phase_jump / count

        singular_near = np.zeros_like(argF, dtype=bool)
        dx = X[0, 1] - X[0, 0]
        radius2 = (2.5 * dx) ** 2
        for xc, yc in zip(Xc[zero_mask], Yc[zero_mask]):
            singular_near |= (X - xc) ** 2 + (Y - yc) ** 2 < radius2
        for xc, yc in zip(Xc[pole_mask], Yc[pole_mask]):
            singular_near |= (X - xc) ** 2 + (Y - yc) ** 2 < radius2

        branch_cut_mask = (
            np.isfinite(phase_jump_score)
            & (phase_jump_score > 1.8)
            & (~singular_near)
        )

        self._diag_cache = {
            'X': X,
            'Y': Y,
            'F': F,
            'finite_mask': finite_mask,
            'logabs': logabs,
            'argF': argF,
            'Xc': Xc,
            'Yc': Yc,
            'winding': winding,
            'zero_mask': zero_mask,
            'pole_mask': pole_mask,
            'phase_jump_score': phase_jump_score,
            'branch_cut_mask': branch_cut_mask,
        }
        return self._diag_cache

    def plot_all(self, figsize=(14, 10), diagnostics=None):
        d = self.compute_diagnostics() if diagnostics is None else diagnostics
        X, Y = d['X'], d['Y']
        Xc, Yc = d['Xc'], d['Yc']

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        ax = axes[0, 0]
        im = ax.pcolormesh(X, Y, d['logabs'], shading='auto', cmap='viridis')
        ax.scatter(Xc[d['pole_mask']], Yc[d['pole_mask']], s=35, marker='x', label='likely poles')
        ax.scatter(Xc[d['zero_mask']], Yc[d['zero_mask']], s=25, marker='o', facecolors='none', label='likely zeros')
        ax.set_title(r'$\log_{10}|f(z)|$')
        ax.set_xlabel('Re z')
        ax.set_ylabel('Im z')
        ax.legend(loc='upper right')
        plt.colorbar(im, ax=ax)

        ax = axes[0, 1]
        im = ax.pcolormesh(X, Y, d['argF'], shading='auto', cmap='hsv', vmin=-np.pi, vmax=np.pi)
        ax.scatter(Xc[d['pole_mask']], Yc[d['pole_mask']], s=35, marker='x')
        ax.scatter(Xc[d['zero_mask']], Yc[d['zero_mask']], s=25, marker='o', facecolors='none')
        ax.set_title(r'$\arg f(z)$')
        ax.set_xlabel('Re z')
        ax.set_ylabel('Im z')
        plt.colorbar(im, ax=ax)

        ax = axes[1, 0]
        im = ax.pcolormesh(Xc, Yc, d['winding'], shading='auto', cmap='coolwarm', vmin=-1.5, vmax=1.5)
        ax.set_title('local winding number')
        ax.set_xlabel('Re z')
        ax.set_ylabel('Im z')
        plt.colorbar(im, ax=ax)

        ax = axes[1, 1]
        background = np.where(np.isfinite(d['phase_jump_score']), d['phase_jump_score'], np.nan)
        im = ax.pcolormesh(X, Y, background, shading='auto', cmap='magma')
        ax.contour(X, Y, d['branch_cut_mask'].astype(float), levels=[0.5], linewidths=1.5)
        ax.scatter(Xc[d['pole_mask']], Yc[d['pole_mask']], s=35, marker='x', label='likely poles')
        ax.scatter(Xc[d['zero_mask']], Yc[d['zero_mask']], s=25, marker='o', facecolors='none', label='likely zeros')
        ax.set_title('branch-cut candidates from phase jumps')
        ax.set_xlabel('Re z')
        ax.set_ylabel('Im z')
        ax.legend(loc='upper right')
        plt.colorbar(im, ax=ax)

        plt.tight_layout()
        plt.show()

    def print_candidates(self, max_items=20, diagnostics=None):
        d = self.compute_diagnostics() if diagnostics is None else diagnostics
        Xc, Yc = d['Xc'], d['Yc']

        poles = list(zip(Xc[d['pole_mask']], Yc[d['pole_mask']], d['winding'][d['pole_mask']]))
        zeros = list(zip(Xc[d['zero_mask']], Yc[d['zero_mask']], d['winding'][d['zero_mask']]))

        def deduplicate(points, tol):
            kept = []
            for x, y, w in points:
                good = True
                for x0, y0, _ in kept:
                    if (x - x0) ** 2 + (y - y0) ** 2 < tol ** 2:
                        good = False
                        break
                if good:
                    kept.append((x, y, w))
            return kept

        grid_step = (self.xlim[1] - self.xlim[0]) / (self.resolution - 1)
        poles = deduplicate(poles, 2.5 * grid_step)
        zeros = deduplicate(zeros, 2.5 * grid_step)

        print('\nLikely poles:')
        for x, y, w in poles[:max_items]:
            print(f'  z ~ {x:+.6f} {y:+.6f}i   winding ~ {w:+.3f}')
        if not poles:
            print('  none detected on this grid')

        print('\nLikely zeros:')
        for x, y, w in zeros[:max_items]:
            print(f'  z ~ {x:+.6f} {y:+.6f}i   winding ~ {w:+.3f}')
        if not zeros:
            print('  none detected on this grid')

        n_branch = int(np.count_nonzero(d['branch_cut_mask']))
        print(f'\nNumber of branch-cut candidate pixels: {n_branch}')
        print('Interpretation: connected chains of these pixels often trace the chosen branch cut.')


if __name__ == '__main__':
    def f(z):
        return mp.sqrt((z-1)*(z+1)) * (z + 1) / (z - 1)

    viewer = ComplexStructureViewer(
        f,
        xlim=(-2, 2),
        ylim=(-2, 2),
        resolution=220,
    )

    diagnostics = viewer.compute_diagnostics()
    viewer.print_candidates(diagnostics=diagnostics)
    viewer.plot_all(diagnostics=diagnostics)
