"""
Splines Hermite Cubiques:
* L'utilsateur peut choisir la parametrisation qu'il veut :chordale, centridède ou equidistante 
* aussi , peut choisir le parametre de tension c 
* mais peut pas choisir les valeurs de m0 et mN , car ils sont calculées automatiquement 
c'est mieux de coté stabilité numérique
* Choix initial entre bessel_tangents et Cardinal Splines
* Après reset, on peut rechoisir la méthode
"""
from tkinter import *
import numpy as np


class HermiteSplineEditor(Tk):
    def __init__(self):
        super().__init__()

        self.title("Splines Hermite Cubiques")
        self.geometry("1000x800")

        self._selected_data = {"x": 0, "y": 0, 'item': None}
        self.radius = 8           #rayon des points   
        
        # Paramètres Hermite (initialisés après choix)
        self.tangent_method = None
        self.tension = DoubleVar(value=0.5)
        self.parametrization = StringVar(value="equidistant")

        # Références aux widgets
        self.canvas = None
        self.control_panel = None

        # Premier choix de méthode
        self.choose_method_and_setup()





    def choose_method_and_setup(self):
        """Choisir la méthode et setup l'interface"""
        method = self.choose_tangent_method()
        if method is None:
            self.destroy()
            return
            
        self.tangent_method = method
        self.setup_canvas()
        self.setup_control_panel()




    def choose_tangent_method(self):
        """Fenêtre de choix de la méthode"""
        choice_window = Toplevel(self)
        choice_window.title("Choix de la méthode")
        choice_window.geometry("350x150")
        choice_window.transient(self)
        choice_window.grab_set()
        
        Label(choice_window, text="Choisissez la méthode de calcul des tangentes:", 
              pady=10).pack()
        
        method = StringVar(value="cardinal")
        
        def confirm_choice():
            choice_window.destroy()
            
        frame = Frame(choice_window)
        frame.pack(pady=10)
        
        Radiobutton(frame, text="Cardinal Splines (avec paramètre c)", 
                   variable=method, value="cardinal").pack(anchor=W)
        Radiobutton(frame, text="Bessel (interpolation quadratique)", 
                   variable=method, value="bessel").pack(anchor=W)
        
        Button(choice_window, text="Confirmer", command=confirm_choice).pack(pady=10)
        
        self.wait_window(choice_window)
        return method.get()




    def setup_canvas(self):
        """Setup du canvas de dessin"""
        if self.canvas:
            self.canvas.destroy()
            
        self.canvas = Canvas(self, bd=2, cursor="plus", bg="#fbfbfb")
        self.canvas.pack(fill=BOTH, expand=True, padx=2, pady=2)

        self.canvas.bind('<Button-1>', self.handle_canvas_click)
        self.canvas.tag_bind("control_points", "<ButtonRelease-1>", self.handle_drag_stop)
        self.canvas.bind("<B1-Motion>", self.handle_drag)





    def setup_control_panel(self):
        """Setup du panneau de contrôle"""
        if self.control_panel:
            self.control_panel.destroy()
            
        self.control_panel = Frame(self, relief=RAISED, bg="#e1e1e1")
        self.control_panel.pack(fill=X, padx=2, pady=2)

        # Affichage de la méthode choisie
        method_frame = Frame(self.control_panel, bg="#e1e1e1")
        method_frame.pack(fill=X, padx=5, pady=2)
        
        method_name = "Cardinal Splines" if self.tangent_method == "cardinal" else "Bessel"
        Label(method_frame, text=f"Méthode: {method_name}", bg="#e1e1e1", 
              font=("Arial", 10, "bold")).pack(side=LEFT)

        # Curseur de tension (seulement pour Cardinal Splines)
        if self.tangent_method == "cardinal":
            tension_frame = Frame(self.control_panel, bg="#e1e1e1")
            tension_frame.pack(fill=X, padx=5, pady=2)
            
            Label(tension_frame, text="Tension c:", bg="#e1e1e1").pack(side=LEFT)
            Scale(tension_frame, from_=0, to=1, resolution=0.1, orient=HORIZONTAL,
                  variable=self.tension, bg="#e1e1e1", length=200,
                  command=lambda x: self.update_display()).pack(side=LEFT)

        # Paramétrisation
        param_frame = Frame(self.control_panel, bg="#e1e1e1")
        param_frame.pack(fill=X, padx=5, pady=2)
        
        Label(param_frame, text="Paramétrisation:", bg="#e1e1e1").pack(side=LEFT)
        for text, value in [("Équidistante", "equidistant"), 
                           ("Chordale", "chordal"), 
                           ("Centripète", "centripetal")]:
            Radiobutton(param_frame, text=text, value=value,
                       variable=self.parametrization, bg="#e1e1e1",
                       command=self.update_display).pack(side=LEFT)

        # Bouton reset
        Button(self.control_panel, text="Reset", command=self.reset_all).pack(side=RIGHT, padx=5)







###---------------------------------------------------------------------------###
    # === MÉTHODES HERMITE ===
    
    def compute_parameters(self, points):
        """Calcule les paramètres selon la méthode choisie"""
        n = len(points)
        u = [0.0]
        
        for i in range(1, n):
            if self.parametrization.get() == "equidistant":
                u.append(u[-1] + 1)
            elif self.parametrization.get() == "chordal":
                dist = np.linalg.norm(np.array(points[i]) - np.array(points[i-1]))
                u.append(u[-1] + dist)
            else:                               # centripetal
                dist = np.linalg.norm(np.array(points[i]) - np.array(points[i-1]))
                u.append(u[-1] + np.sqrt(dist))
        
        return np.array(u)
    



    
    def compute_hermite_tangents(self, points, u):
        """Calcule les tangentes avec la méthode choisie"""
        if self.tangent_method == "cardinal":
            return self.cardinal_tangents(points, u)
        else:
            return self.bessel_tangents(points, u)





    def cardinal_tangents(self, points, u):
        """Calcule les tangentes avec Cardinal Splines"""
        n = len(points)
        if n < 2:
            return []
        
        tangents = []
        c = self.tension.get()
        
        # Tangentes intérieures
        for k in range(1, n-1):
            tangent = (1 - c) * (np.array(points[k+1]) - np.array(points[k-1])) / (u[k+1] - u[k-1])
            tangents.append(tangent)
        
        # Tangentes aux extrémités (naturelles)
        if n >= 2:
            m0 = 2 * (np.array(points[1]) - np.array(points[0])) / (u[1] - u[0]) - tangents[0]
            mN = 2 * (np.array(points[-1]) - np.array(points[-2])) / (u[-1] - u[-2]) - tangents[-1]
            return [m0] + tangents + [mN]
        else:
            return [np.array([0,0])] * n




    def bessel_tangents(self, points, u):
        """Méthode de Bessel - interpolation quadratique"""
        n = len(points)
        if n < 2:
            return [np.array([0,0]) for _ in range(n)]
        
        tangents = []
        for k in range(n):
            if k == 0:
                tangent = (np.array(points[1]) - np.array(points[0])) / (u[1] - u[0])
            elif k == n-1:
                tangent = (np.array(points[-1]) - np.array(points[-2])) / (u[-1] - u[-2])
            else:
                # Moyenne pondérée des pentes
                w1 = (u[k+1] - u[k]) / (u[k+1] - u[k-1])
                w2 = (u[k] - u[k-1]) / (u[k+1] - u[k-1])
                tangent = w1 * (np.array(points[k]) - np.array(points[k-1]))/(u[k] - u[k-1]) + \
                         w2 * (np.array(points[k+1]) - np.array(points[k]))/(u[k+1] - u[k])
            tangents.append(tangent)
        
        return tangents




    def hermite_spline(self, points, T):
        """Spline Hermite cubique complète"""
        if len(points) < 2:
            return np.array(points)
        
        u = self.compute_parameters(points)
        tangents = self.compute_hermite_tangents(points, u)
        
        result = []
        for t_global in T:
            # Trouver le segment correspondant
            segment_idx = 0
            t_local = t_global * (len(points) - 1)
            
            for k in range(len(points)-1):
                if k <= t_local < k+1:
                    segment_idx = k
                    break
                elif k == len(points)-2 and t_local >= k+1:
                    segment_idx = k
            
            t = t_local - segment_idx  # t ∈ [0,1] dans le segment
            
            # Points de contrôle Bézier pour ce segment
            h = u[segment_idx+1] - u[segment_idx]
            P0 = np.array(points[segment_idx])
            P1 = np.array(points[segment_idx+1])
            m0 = tangents[segment_idx]
            m1 = tangents[segment_idx+1]
            
            b0 = P0
            b1 = P0 + (h/3) * m0
            b2 = P1 - (h/3) * m1
            b3 = P1
            
            # Courbe de Bézier
            point = (1-t)**3 * b0 + 3*t*(1-t)**2 * b1 + 3*t**2*(1-t) * b2 + t**3 * b3
            result.append(point)
        
        return np.array(result)





###----------------------------------------------------------------------------###
    # === MÉTHODES D'INTERFACE ===
    
    def get_points(self):
        points = []
        if self.canvas:
            for item in self.canvas.find_withtag("control_points"):
                coords = self.canvas.coords(item)
                points.append([float(coords[0] + self.radius), float(coords[1] + self.radius)])
        return points


    def create_point(self, x, y):
        if not self.canvas:
            return None
        item = self.canvas.create_oval(x - self.radius, y - self.radius,
                                      x + self.radius, y + self.radius,
                                      outline="red", fill="red", tags="control_points")
        return item


    def draw_curve(self):
        if not self.canvas:
            return
        self.canvas.delete("curve")
        points = self.get_points()
        if len(points) <= 1:
            return

        # Générer la courbe Hermite
        T = np.linspace(0, 1, 100)
        curve = self.hermite_spline(points, T)

        # Dessiner la courbe
        for i in range(len(curve) - 1):
            self.canvas.create_line(curve[i, 0], curve[i, 1],
                                   curve[i + 1, 0], curve[i + 1, 1],
                                   fill="green", width=3, tags="curve")


    def find_closest_point(self, x, y, radius):
        if not self.canvas:
            return (None, [0, 0], float("inf"))
        distances = []
        for item in self.canvas.find_withtag("control_points"):
            coords = self.canvas.coords(item)
            dist = (x - coords[0])**2 + (y - coords[1])**2
            if dist <= radius**2:
                distances.append((item, coords, dist))
        return min(distances, default=(None, [0, 0], float("inf")), key=lambda p: p[2])


    def handle_canvas_click(self, event):
        if not self.canvas:
            return
        if self._selected_data['item'] is not None:
            self.canvas.itemconfig(self._selected_data['item'], fill='red')
            self._selected_data['item'] = None

        # Ajouter un point
        item = self.create_point(event.x, event.y)
        self._selected_data['item'] = item
        self.update_display()


    def handle_drag_stop(self, event):
        if self._selected_data['item'] is not None:
            if self.canvas:
                self.canvas.itemconfig(self._selected_data['item'], fill='red')
            self._selected_data['item'] = None


    def handle_drag(self, event):
        if self._selected_data['item'] is None or not self.canvas:
            return

        # Trouver le point le plus proche
        item, coords, _ = self.find_closest_point(event.x, event.y, 3 * self.radius)
        if item is not None and item == self._selected_data['item']:
            # Déplacer le point
            self.canvas.coords(item, 
                              event.x - self.radius, event.y - self.radius,
                              event.x + self.radius, event.y + self.radius)
            self.update_display()


    def update_display(self, event=None):
        self.draw_curve()


    def reset_all(self):
        """Reset complet avec rechoisir la méthode"""
        # Fermer toutes les fenêtres Toplevel existantes
        for widget in self.winfo_children():
            if isinstance(widget, Toplevel):
                widget.destroy()
        
        # Réinitialiser les données
        self._selected_data = {"x": 0, "y": 0, 'item': None}
        
        # Rechoisir la méthode et refaire le setup
        self.choose_method_and_setup()

###---------------------------------------------------------------------###

if __name__ == "__main__":
    app = HermiteSplineEditor()
    app.mainloop()