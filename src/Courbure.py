import numpy as np
import matplotlib.pyplot as plt

class Courbure:
    def __init__(self):
        # Points pour former un cœur
        self.points = np.array([
            [1, 0],
            [1, 2],
            [2, 3],
            [3, 1],
            [4, 3],
            [5, 2],
            [5, 0], 
            [3, -2],
            [1, 0]
        ])
        
        # Paramétrisation équidistante
        self.u = np.arange(len(self.points))

    def run_hermite(self, c=0.5, methode="cardinal"):
        """Spline Hermite cubique avec calcul de courbure"""
        n = len(self.points)
        if n < 2:
            return np.array([]), np.array([])
        
        # 1. Calcul des tangentes
        if methode == "cardinal":
            tangents = self._compute_cardinal_tangents(c)
        else:  # bessel
            tangents = self.bessel_tangents()  # Correction: enlever u en paramètre
        
        # 2. Génération de la courbe segment par segment
        curve_points = []
        curvatures = []
        u_global = []
        
        for i in range(n - 1):
            Pk = self.points[i]
            Pk1 = self.points[i + 1]
            mk = tangents[i]
            mk1 = tangents[i + 1]
            hk = self.u[i + 1] - self.u[i]
            
            # Génération du segment
            segment_points, segment_curvatures, segment_u = self._compute_hermite_segment(
                Pk, Pk1, mk, mk1, hk, self.u[i]
            )
            
            curve_points.extend(segment_points)
            curvatures.extend(segment_curvatures)
            u_global.extend(segment_u)
        
        return np.array(curve_points), np.array(curvatures)

    def _compute_cardinal_tangents(self, c):
        """Calcule les tangentes avec Cardinal splines"""
        n = len(self.points)
        tangents = []
        
        # Points intérieurs
        for k in range(1, n - 1):
            vector = self.points[k + 1] - self.points[k - 1]
            delta_u = self.u[k + 1] - self.u[k - 1]
            tangents.append((1 - c) * vector / delta_u)
   
        # Points d'extrémités
        tangents.insert(0, (1 - c) * (self.points[1] - self.points[0]) / (self.u[1] - self.u[0]))
        tangents.append((1 - c) * (self.points[-1] - self.points[-2]) / (self.u[-1] - self.u[-2]))
        
        return tangents

    def bessel_tangents(self):
        """Méthode de Bessel - interpolation quadratique"""
        points = self.points
        u = self.u
        n = len(points)
        
        tangents = []
        for k in range(n):
            if k == 0:
                # Premier point: différence forward
                tangent = (points[1] - points[0]) / (u[1] - u[0])
            elif k == n-1:
                # Dernier point: différence backward  
                tangent = (points[-1] - points[-2]) / (u[-1] - u[-2])
            else:
                # Points intérieurs: moyenne pondérée
                w1 = (u[k+1] - u[k]) / (u[k+1] - u[k-1])
                w2 = (u[k] - u[k-1]) / (u[k+1] - u[k-1])
                tangent = (w1 * (points[k] - points[k-1])/(u[k] - u[k-1]) + 
                          w2 * (points[k+1] - points[k])/(u[k+1] - u[k]))
            tangents.append(tangent)
        
        return tangents

    def _compute_hermite_segment(self, Pk, Pk1, mk, mk1, hk, u_start, num_points=100):
        """Calcule un segment de spline Hermite avec sa courbure"""
        t_values = np.linspace(0, 1, num_points)
        segment_points = []
        segment_curvatures = []
        segment_u = []
        
        # Coefficients polynomiaux
        ax, bx, cx, dx = self._hermite_to_polynomial(Pk[0], Pk1[0], mk[0], mk1[0], hk)
        ay, by, cy, dy = self._hermite_to_polynomial(Pk[1], Pk1[1], mk[1], mk1[1], hk)
        
        for t in t_values:
            # Point sur la courbe
            x = ax*t**3 + bx*t**2 + cx*t + dx
            y = ay*t**3 + by*t**2 + cy*t + dy
            segment_points.append([x, y])
            
            # Courbure
            curvature = self._compute_curvure(t, ax, bx, cx, ay, by, cy, hk)
            segment_curvatures.append(curvature)
            
            # Paramètre global
            u = u_start + hk * t
            segment_u.append(u)
        
        return segment_points, segment_curvatures, segment_u

    def _hermite_to_polynomial(self, pk, pk1, mk, mk1, hk):
        """Convertit la forme Hermite en forme polynomiale standard"""
        a = 2*pk - 2*pk1 + hk*mk + hk*mk1
        b = -3*pk + 3*pk1 - 2*hk*mk - hk*mk1
        c = hk*mk
        d = pk
        return a, b, c, d

    def _compute_curvure(self, t, ax, bx, cx, ay, by, cy, hk):
        """Calcule la courbure au paramètre t"""
        xp = 3*ax*t**2 + 2*bx*t + cx
        yp = 3*ay*t**2 + 2*by*t + cy
        xpp = 6*ax*t + 2*bx
        ypp = 6*ay*t + 2*by
        
        numerator = xp * ypp - yp * xpp
        denominator = (xp**2 + yp**2)**1.5
        
        if abs(denominator) < 1e-12:
            return 0.0
        else:
            return numerator / denominator * hk

    def compare_all(self):
        """Compare différentes configurations Hermite"""
        fig, axes = plt.subplots(2, 3, figsize=(16, 8))
        
        # Configurations à comparer
        configurations = [
            ('Cardinal c=0.3', self.run_hermite(0.3, "cardinal")),
            ('Cardinal c=0.6', self.run_hermite(0.7, "cardinal")),
            ('Bessel', self.run_hermite(0.5, "bessel"))       # c ignoré pour Bessel
        ]
        
        for i, (name, (curve, curvature)) in enumerate(configurations):
            if i >= 4:  # Sécurité
                break
                
            # Courbe
            axes[0, i].plot(curve[:, 0], curve[:, 1], 'b-', linewidth=2)
            axes[0, i].plot(self.points[:, 0], self.points[:, 1], 'ro-', alpha=0.6)
            axes[0, i].set_title(name)
            axes[0, i].grid(True)
            axes[0, i].set_aspect('equal')
            
            # Courbure
            axes[1, i].plot(curvature, 'g-', linewidth=1)
            axes[1, i].set_title(f'Courbure {name}')
            axes[1, i].grid(True)
            axes[1, i].set_ylim(-5, 5)  # Même échelle
        
        plt.tight_layout()
        plt.show()

# Test
experiment = Courbure()
experiment.compare_all()