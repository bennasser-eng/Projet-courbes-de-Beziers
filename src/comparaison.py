import numpy as np
import matplotlib.pyplot as plt


class ComparisonExperiment:
    def __init__(self):
        # Points pour former un objet:
        """self.points = np.array([
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
        


        
        self.points = np.array([
            [0,0],
            [4,2],
            [3,4],
            [2,5],
            [1,4],
            [0,0]
        ])

        
        self.points = np.array([
            [0, 0],
            [1, 2],
            [3, 5],
            [5, 1],
            [7, 6],
            [8, 3],
            [9, 7],
            [7, 4],
            [5, 8],
            [3, 3],
            [2, 6],
            [1, 1],
            [0, 0]
        ])
        

        self.points = np.array([
            [0, 0],
            [1, 4],
            [3, 10],
            [5, 1],
            [7, 0],
            [8, 3],
            [9, 7],
            [7, 4],
            [5, 8],
            [3, 3],
            [2, 3],
            [1, 1],
            [2, 4],
            [1, 10],
            [0, 0]
        ])"""

        self.points = np.array([
            [0,0],
            [3,2],
            [2,1],
            [1,3]
        ])


        # Paramétrisation équidistante
        self.u = np.arange(len(self.points))



    def _compute_numerical_curvature(self, curve_points, u_values):
        """Calcule la courbure par différences finies"""
        n = len(curve_points)
        if n < 3:
            return np.zeros(n)
        
        curvatures = np.zeros(n)
        
        # Différences premières et secondes
        dx_du = np.gradient(curve_points[:, 0], u_values)
        dy_du = np.gradient(curve_points[:, 1], u_values)
        d2x_du2 = np.gradient(dx_du, u_values)
        d2y_du2 = np.gradient(dy_du, u_values)
        
        # Courbure
        numerator = dx_du * d2y_du2 - dy_du * d2x_du2
        denominator = (dx_du**2 + dy_du**2)**1.5
        
        # Éviter les divisions par zéro
        mask = np.abs(denominator) > 1e-12
        curvatures[mask] = numerator[mask] / denominator[mask]
        
        return curvatures



    def run_hermite(self, c=0.5):
        """Spline Hermite cubique avec calcul de courbure"""
        n = len(self.points)
        if n < 2:
            return np.array([]), np.array([])
        
        # 1. Calcul des tangentes avec Cardinal splines
        tangents = self._compute_cardinal_tangents(c)
        
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
   
        # Les points d'extrémités : Différence finie forward/backward
        tangents.insert(0, (1 - c) * (self.points[1] - self.points[0]) / (self.u[1] - self.u[0]))
        tangents.append((1 - c) * (self.points[-1] - self.points[-2]) / (self.u[-1] - self.u[-2]))
        
        return tangents


    def _compute_hermite_segment(self, Pk, Pk1, mk, mk1, hk, u_start, num_points=100):
        """Calcule un segment de spline Hermite avec sa courbure"""
        t_values = np.linspace(0, 1, num_points)
        segment_points = []
        segment_curvatures = []
        segment_u = []
        
        # Coefficients polynomiaux pour calcul efficace
        ax, bx, cx, dx = self._hermite_to_polynomial(Pk[0], Pk1[0], mk[0], mk1[0], hk)
        ay, by, cy, dy = self._hermite_to_polynomial(Pk[1], Pk1[1], mk[1], mk1[1], hk)
        
        for t in t_values:
            # Point sur la courbe
            x = ax*t**3 + bx*t**2 + cx*t + dx
            y = ay*t**3 + by*t**2 + cy*t + dy
            segment_points.append([x, y])
            
            # Courbure
            curvature = self._compute_curvature(t, ax, bx, cx, ay, by, cy, hk)
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


    def _compute_curvature(self, t, ax, bx, cx, ay, by, cy, hk):
        """Calcule la courbure au paramètre t"""
        # Dérivées premières par rapport à t
        xp = 3*ax*t**2 + 2*bx*t + cx
        yp = 3*ay*t**2 + 2*by*t + cy
        
        # Dérivées secondes par rapport à t
        xpp = 6*ax*t + 2*bx
        ypp = 6*ay*t + 2*by
        
        # Courbure
        numerator = xp * ypp - yp * xpp
        denominator = (xp**2 + yp**2)**1.5
        
        if abs(denominator) < 1e-12:
            return 0.0  # Évite la division par zéro
        else:
            return numerator / denominator * hk 
        



###-------------------------------------------------------------###

    def run_lagrange(self):
        """Interpolation polynomiale de Lagrange (version optimisée)"""
        
        n = len(self.points)
        if n < 2:
            return np.array([]), np.array([])
        
        # Génération de points d'évaluation
        u_dense = np.linspace(self.u[0], self.u[-1], min(500, 10 * n))
        
        # Interpolation paramétrique
        x_interp = self._lagrange_interpolation(u_dense, self.u, self.points[:, 0])
        y_interp = self._lagrange_interpolation(u_dense, self.u, self.points[:, 1])
        
        curve_points = np.column_stack([x_interp, y_interp])
        
        # Calcul de la courbure
        curvatures = self._compute_numerical_curvature(curve_points, u_dense)
        
        return curve_points, curvatures


    def _lagrange_interpolation(self, u_eval, u_nodes, values):
        """Interpolation de Lagrange vectorisée"""
        n = len(u_nodes)
        result = np.zeros_like(u_eval)
        
        for i in range(n):
            # Calcul du polynôme de base L_i(u)
            L_i = np.ones_like(u_eval)
            for j in range(n):
                if i != j:
                    L_i *= (u_eval - u_nodes[j]) / (u_nodes[i] - u_nodes[j])
            result += values[i] * L_i
        
        return result





###-----------------------------------------------------------------------###

    def run_c2_spline(self):
        """Spline cubique C² naturelle avec paramétrisation équidistante"""
        
        n = len(self.points)
        if n < 3:
            return np.array([]), np.array([])
        
        # Résolution du système tridiagonal pour les dérivées secondes
        second_derivs = self._solve_natural_spline_system()
        
        # Génération de la courbe
        curve_points = []
        curvatures = []
        u_global = []
        
        for i in range(n - 1):
            segment_points, segment_curvatures, segment_u = self._compute_c2_segment(
                i, second_derivs
            )
            curve_points.extend(segment_points)
            curvatures.extend(segment_curvatures)
            u_global.extend(segment_u)
        
        return np.array(curve_points), np.array(curvatures)


    def _solve_natural_spline_system(self):
        """Résout le système tridiagonal pour les splines naturelles C²"""
        n = len(self.points)
        h = 1.0  # paramétrisation équidistante
        
        # Construction du système Ax = b
        A = np.zeros((n, n))
        b_x = np.zeros(n)
        b_y = np.zeros(n)
        
        # Conditions naturelles aux extrémités
        A[0, 0] = 1.0
        A[-1, -1] = 1.0
        
        # Équations pour les points intérieurs
        for i in range(1, n - 1):
            A[i, i - 1] = h
            A[i, i] = 4 * h
            A[i, i + 1] = h
            
            b_x[i] = 6 * (self.points[i + 1, 0] - 2 * self.points[i, 0] + self.points[i - 1, 0]) / h
            b_y[i] = 6 * (self.points[i + 1, 1] - 2 * self.points[i, 1] + self.points[i - 1, 1]) / h
        
        # Résolution des systèmes
        M_x = np.linalg.solve(A, b_x)
        M_y = np.linalg.solve(A, b_y)
        
        return np.column_stack([M_x, M_y])


    def _compute_c2_segment(self, i, second_derivs):
        """Calcule un segment de spline C²"""
        h = 1.0  # paramétrisation équidistante
        t_values = np.linspace(0, 1, 100)
        
        segment_points = []
        segment_curvatures = []
        segment_u = []
        
        # Points et dérivées secondes pour ce segment
        P0 = self.points[i]
        P1 = self.points[i + 1]
        M0 = second_derivs[i]
        M1 = second_derivs[i + 1]
        
        for t in t_values:
            # Forme de Hermite pour les splines cubiques
            point = self._c2_spline_formula(t, P0, P1, M0, M1, h)
            segment_points.append(point)
            
            # Courbure
            curvature = self._c2_curvature(t, P0, P1, M0, M1, h)
            segment_curvatures.append(curvature)
            
            # Paramètre global
            u = self.u[i] + h * t
            segment_u.append(u)  # CORRECTION: cette ligne était mal indentée
    
        return segment_points, segment_curvatures, segment_u


    def _c2_spline_formula(self, t, P0, P1, M0, M1, h):
        """Formule de la spline cubique C² sur un segment"""
        a = (M1 - M0) / (6 * h)
        b = M0 / 2
        c = (P1 - P0) / h - (2 * h * M0 + h * M1) / 6
        d = P0
        
        point = a * (h * t)**3 + b * (h * t)**2 + c * (h * t) + d
        return point


    def _c2_curvature(self, t, P0, P1, M0, M1, h):
        """Calcule la courbure pour une spline C²"""
        # Dérivées premières
        a = (M1 - M0) / (6 * h)
        b = M0 / 2
        c = (P1 - P0) / h - (2 * h * M0 + h * M1) / 6
        
        xp = 3 * a[0] * (h * t)**2 + 2 * b[0] * (h * t) + c[0]
        yp = 3 * a[1] * (h * t)**2 + 2 * b[1] * (h * t) + c[1]
        
        # Dérivées secondes
        xpp = 6 * a[0] * (h * t) + 2 * b[0]
        ypp = 6 * a[1] * (h * t) + 2 * b[1]
        
        # Courbure
        numerator = xp * ypp - yp * xpp
        denominator = (xp**2 + yp**2)**1.5
        
        if abs(denominator) < 1e-12:
            return 0.0
        else:
            return numerator / denominator







###-------------------------------------------------------------###

    def compare_all(self):
        """Compare les 3 méthodes sur les mêmes points"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Méthodes à comparer
        methods = [
            ('Hermite', self.run_hermite()),
            ('Lagrange', self.run_lagrange()), 
            ('C² Spline', self.run_c2_spline())
        ]
        
        for i, (name, (curve, curvature)) in enumerate(methods):
            # Courbe
            axes[0, i].plot(curve[:, 0], curve[:, 1], 'b-', linewidth=2)
            axes[0, i].plot(self.points[:, 0], self.points[:, 1], 'ro-', alpha=0.6)
            axes[0, i].set_title(f'{name} - Courbe')
            axes[0, i].grid(True)
            axes[0, i].set_aspect('equal')
            
            # Courbure
            axes[1, i].plot(curvature, 'g-', linewidth=2)
            axes[1, i].set_title(f'{name} - Courbure')
            axes[1, i].grid(True)
            # Même échelle pour toutes les courbures
            axes[1, i].set_ylim(-10, 10)
        
        plt.tight_layout()
        plt.show()


# Lancement de l'expérience
experiment = ComparisonExperiment()
experiment.compare_all()