"""
author: Anthony Dard + Ahmed Bennasser
description: Extension pour les splines Hermite cubiques
"""
from tkinter import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class CurveEditorWindow(Tk):
    def __init__(self, compute_algorithms) -> None:
        super().__init__()

        self.title("Splines Hermite Cubiques - Éditeur de Courbes")
        self.geometry("1024x720")

        self._selected_data = {"x": 0, "y": 0, 'item': None}
        self.radius = 5
        self.curve = None
        self.compute_algorithms = compute_algorithms
        
        # Nouveaux paramètres pour les splines Hermite
        self.tension = DoubleVar(value=0.5)
        self.parametrization = StringVar(value="equidistant")
        self.show_control_points = BooleanVar(value=True)
        self.show_tangents = BooleanVar(value=False)
        self.show_curvature = BooleanVar(value=False)

        self.setup_canvas()
        self.setup_panel()
        self.setup_curvature_plot()

    def setup_canvas(self):
        self.graph = Canvas(self, bd=2, cursor="plus", bg="#fbfbfb")
        self.graph.grid(column=0, padx=2, pady=2, rowspan=2, columnspan=2, sticky="nsew")

        self.graph.bind('<Button-1>', self.handle_canvas_click)
        self.graph.tag_bind("control_points", "<ButtonRelease-1>", self.handle_drag_stop)
        self.graph.bind("<B1-Motion>", self.handle_drag)

    def setup_curvature_plot(self):
        """Setup pour l'affichage de la courbure"""
        self.fig, (self.ax_curve, self.ax_curvature) = plt.subplots(2, 1, figsize=(6, 4))
        self.canvas_plot = FigureCanvasTkAgg(self.fig, self)
        self.canvas_plot.get_tk_widget().grid(row=0, column=3, rowspan=2, padx=2, pady=2, sticky="nsew")

    def setup_panel(self):
        # Right panel for options
        self.frame_pannel = Frame(self, relief=RAISED, bg="#e1e1e1")
        
        # Frames
        self.frame_curve_type = Frame(self.frame_pannel)
        self.frame_edit_type = Frame(self.frame_pannel)
        self.frame_edit_position = Frame(self.frame_pannel)
        self.frame_sliders = Frame(self.frame_pannel)
        self.frame_hermite_params = Frame(self.frame_pannel)
        self.frame_display_options = Frame(self.frame_pannel)

        # Selection of curve type
        curve_types = [algo['name'] for algo in self.compute_algorithms]
        curve_types_val = list(range(len(self.compute_algorithms)))
        self.curve_type = IntVar()
        self.curve_type.set(curve_types_val[0])

        self.radio_curve_buttons = [None] * len(self.compute_algorithms)
        for i in range(len(self.compute_algorithms)):
            self.radio_curve_buttons[i] = Radiobutton(self.frame_curve_type,
                                                      variable=self.curve_type,
                                                      text=curve_types[i],
                                                      value=curve_types_val[i],
                                                      bg="#e1e1e1")
            self.radio_curve_buttons[i].pack(side='left', expand=1)
            self.radio_curve_buttons[i].bind("<ButtonRelease-1>", self.update_display)

        # Paramètres des splines Hermite
        Label(self.frame_hermite_params, text="Tension (c):", bg="#e1e1e1").pack(side=LEFT)
        scale_tension = Scale(self.frame_hermite_params, from_=0, to=1, 
                             resolution=0.1, orient=HORIZONTAL,
                             variable=self.tension, bg="#e1e1e1", length=150)
        scale_tension.pack(side=LEFT)
        scale_tension.bind("<ButtonRelease-1>", self.update_display)

        # Paramétrisation
        Label(self.frame_hermite_params, text="Paramétrisation:", bg="#e1e1e1").pack(side=LEFT)
        parametrizations = [("Équidistante", "equidistant"), 
                          ("Chordale", "chordal"), 
                          ("Centripète", "centripetal")]
        for text, value in parametrizations:
            Radiobutton(self.frame_hermite_params, text=text, value=value,
                       variable=self.parametrization, bg="#e1e1e1").pack(side=LEFT)
        self.parametrization.trace('w', lambda *args: self.update_display())

        # Options d'affichage
        Checkbutton(self.frame_display_options, text="Points de contrôle", 
                   variable=self.show_control_points, bg="#e1e1e1",
                   command=self.update_display).pack(side=LEFT)
        Checkbutton(self.frame_display_options, text="Tangentes", 
                   variable=self.show_tangents, bg="#e1e1e1",
                   command=self.update_display).pack(side=LEFT)
        Checkbutton(self.frame_display_options, text="Courbure", 
                   variable=self.show_curvature, bg="#e1e1e1",
                   command=self.update_curvature_display).pack(side=LEFT)

        # Setup des frames existants
        self.setup_existing_frames()
        
        # Packing des frames
        self.frame_curve_type.pack(fill=X, padx=5, pady=2)
        self.frame_hermite_params.pack(fill=X, padx=5, pady=2)
        self.frame_display_options.pack(fill=X, padx=5, pady=2)
        self.frame_edit_type.pack(fill=X, padx=5, pady=2)
        self.frame_edit_position.pack(fill=X, padx=5, pady=2)
        self.frame_sliders.pack(fill=X, padx=5, pady=2)

        self.button_reset = Button(self.frame_pannel, text="Reset", command=self.reset_all)
        self.button_reset.pack(side=BOTTOM, fill="x")

        self.frame_pannel.grid(row=0, column=2, padx=2, pady=2, rowspan=2, sticky="nswe")

    def setup_existing_frames(self):
        """Setup des frames d'édition"""
        # Selection of edit mode
        edit_types = ['Add', 'Remove', 'Drag', 'Select']
        edit_types_val = ["add", "remove", "drag", "select"]
        self.edit_types = StringVar()
        self.edit_types.set(edit_types_val[0])

        self.radio_edit_buttons = [None] * 4
        for i in range(4):
            self.radio_edit_buttons[i] = Radiobutton(self.frame_edit_type,
                                                     variable=self.edit_types,
                                                     text=edit_types[i],
                                                     value=edit_types_val[i],
                                                     bg="#e1e1e1")
            self.radio_edit_buttons[i].pack(side='left', expand=1)
            self.radio_edit_buttons[i].bind("<ButtonRelease-1>", lambda event: self.reset_selection())

        # Edit position of selected point widget
        self.label_pos_x = Label(self.frame_edit_position, text='x: ')
        self.label_pos_y = Label(self.frame_edit_position, text='y: ')
        self.pos_x = StringVar()
        self.pos_y = StringVar()
        self.entry_position_x = Entry(self.frame_edit_position, textvariable=self.pos_x)
        self.entry_position_y = Entry(self.frame_edit_position, textvariable=self.pos_y)
        self.label_pos_x.grid(row=0, column=0)
        self.entry_position_x.grid(row=0, column=1)
        self.label_pos_y.grid(row=1, column=0)
        self.entry_position_y.grid(row=1, column=1)

        self.entry_position_x.bind("<FocusOut>", self.update_pos)
        self.entry_position_x.bind("<KeyPress-Return>", self.update_pos)
        self.entry_position_x.bind("<KeyPress-KP_Enter>", self.update_pos)
        self.entry_position_y.bind("<FocusOut>", self.update_pos)
        self.entry_position_y.bind("<KeyPress-Return>", self.update_pos)
        self.entry_position_y.bind("<KeyPress-KP_Enter>", self.update_pos)

        # Slider for parameter update
        self.label_resolution = Label(self.frame_sliders, text="Résolution: ")
        self.slider_resolution = Scale(self.frame_sliders, from_=5, to=500, orient=HORIZONTAL, bg="#e1e1e1")
        self.slider_resolution.set(100)
        self.label_resolution.grid(row=0, column=0)
        self.slider_resolution.grid(row=0, column=1)
        self.slider_resolution.bind("<ButtonRelease-1>", lambda event: self.update_display())

    # === MÉTHODES D'INTERACTION (ORIGINALES) ===
    
    def get_points(self):
        points = []
        for item in self.graph.find_withtag("control_points"):
            coords = self.graph.coords(item)
            points.append([float(coords[0] + self.radius), float(coords[1] + self.radius)])
        return points

    def create_point(self, x, y, color, tags="control_points"):
        item = self.graph.create_oval(x - self.radius, y - self.radius,
                                     x + self.radius, y + self.radius,
                                     outline=color, fill=color, tags=tags)
        return item

    def draw_polygon(self):
        self.graph.delete("control_polygon")
        points = self.get_points()
        for i in range(0, len(points) - 1):
            self.graph.create_line(points[i][0], points[i][1],
                                   points[i + 1][0], points[i + 1][1],
                                   fill="blue", tags="control_polygon")

    def draw_curve(self):
        self.graph.delete("curve")
        points = self.get_points()
        if len(points) <= 1:
            return

        # Utiliser l'algorithme sélectionné
        algo = self.compute_algorithms[self.curve_type.get()]['algo']
        self.curve = algo(np.array(points), np.linspace(0, 1, self.slider_resolution.get()))

        for i in range(0, self.curve.shape[0] - 1):
            self.graph.create_line(self.curve[i, 0], self.curve[i, 1],
                                   self.curve[i + 1, 0], self.curve[i + 1, 1],
                                   fill="green", width=3, tags="curve")

    def find_closest_with_tag(self, x, y, radius, tag):
        distances = []
        for item in self.graph.find_withtag(tag):
            c = self.graph.coords(item)
            d = (x - c[0])**2 + (y - c[1])**2
            if d <= radius**2:
                distances.append((item, c, d))
        return min(distances, default=(None, [0, 0], float("inf")), key=lambda p: p[2])

    def reset_selection(self):
        if self._selected_data['item'] is not None:
            self.graph.itemconfig(self._selected_data['item'], fill='red')
        self._selected_data['item'] = None
        self._selected_data["x"] = 0
        self._selected_data["y"] = 0

    def handle_canvas_click(self, event):
        self.reset_selection()

        if self.edit_types.get() == "add":
            item = self.create_point(event.x, event.y, "red")
            self.update_pos_entry(item)
            self.draw_polygon()
            self.update_display()

        elif self.edit_types.get() == "remove":
            self._selected_data['item'], coords, _ = self.find_closest_with_tag(
                event.x, event.y, 3 * self.radius, "control_points")
            if self._selected_data['item'] is not None:
                self.graph.delete(self._selected_data['item'])
                self.draw_polygon()
                self.update_display()

        elif self.edit_types.get() == "drag":
            self._selected_data['item'], coords, _ = self.find_closest_with_tag(
                event.x, event.y, 3 * self.radius, "control_points")
            if self._selected_data['item'] is not None:
                self._selected_data["x"] = event.x
                self._selected_data["y"] = event.y
                self.graph.move(self._selected_data['item'],
                                event.x - coords[0] - self.radius,
                                event.y - coords[1] - self.radius)

        else:  # select
            self._selected_data['item'], coords, _ = self.find_closest_with_tag(
                event.x, event.y, 3 * self.radius, "control_points")
            if self._selected_data['item'] is not None:
                self.graph.itemconfig(self._selected_data['item'], fill='orange')
                self.update_pos_entry(self._selected_data['item'])

    def handle_drag_stop(self, event):
        if self.edit_types.get() != "drag":
            return
        self.reset_selection()

    def handle_drag(self, event):
        if self.edit_types.get() != "drag" or self._selected_data['item'] is None:
            return

        delta_x = event.x - self._selected_data["x"]
        delta_y = event.y - self._selected_data["y"]
        self.graph.move(self._selected_data['item'], delta_x, delta_y)
        self._selected_data["x"] = event.x
        self._selected_data["y"] = event.y

        self.update_pos_entry(self._selected_data['item'])
        self.draw_polygon()
        self.update_display()

    def update_pos_entry(self, item):
        coords = self.graph.coords(item)
        self.entry_position_x.delete(0, END)
        self.entry_position_x.insert(0, int(coords[0]))
        self.entry_position_y.delete(0, END)
        self.entry_position_y.insert(0, int(coords[1]))

    def update_pos(self, event):
        if self.edit_types.get() != "select" or self._selected_data['item'] is None:
            return
        coords = self.graph.coords(self._selected_data['item'])
        self.graph.move(self._selected_data['item'],
                        float(self.pos_x.get()) - coords[0],
                        float(self.pos_y.get()) - coords[1])
        self.draw_polygon()
        self.update_display()

    # === NOUVELLES MÉTHODES POUR LES SPLINES HERMITE ===

    def reset_all(self):
        self.graph.delete("all")
        self.ax_curve.clear()
        self.ax_curvature.clear()
        self.canvas_plot.draw()

    def compute_parameters(self, points):
        """Calcule les paramètres selon la méthode choisie"""
        n = len(points)
        if n <= 1:
            return np.arange(n)
        
        u = [0.0]
        if self.parametrization.get() == "equidistant":
            for i in range(1, n):
                u.append(u[-1] + 1)
        elif self.parametrization.get() == "chordal":
            for i in range(1, n):
                dist = np.linalg.norm(np.array(points[i]) - np.array(points[i-1]))
                u.append(u[-1] + dist)
        elif self.parametrization.get() == "centripetal":
            for i in range(1, n):
                dist = np.linalg.norm(np.array(points[i]) - np.array(points[i-1]))
                u.append(u[-1] + np.sqrt(dist))
        
        return np.array(u)

    def compute_cardinal_tangents(self, points, u):
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
            tangents = [m0] + tangents + [mN]
        else:
            tangents = [np.array([0,0])] * n
            
        return tangents

    def draw_bezier_control_points(self, points, tangents, u):
        """Dessine les points de contrôle Bézier"""
        if not self.show_control_points.get() or len(points) < 2:
            return
            
        self.graph.delete("bezier_control")
        
        for k in range(len(points)-1):
            h = u[k+1] - u[k]
            b0 = np.array(points[k])
            b1 = b0 + (h/3) * tangents[k]
            b2 = np.array(points[k+1]) - (h/3) * tangents[k+1]
            b3 = np.array(points[k+1])
            
            # Points de contrôle
            self.create_point(b1[0], b1[1], "blue", "bezier_control")
            self.create_point(b2[0], b2[1], "blue", "bezier_control")
            
            # Lignes de contrôle
            self.graph.create_line(b0[0], b0[1], b1[0], b1[1], 
                                 fill="lightblue", tags="bezier_control", dash=(2,2))
            self.graph.create_line(b1[0], b1[1], b2[0], b2[1], 
                                 fill="lightblue", tags="bezier_control", dash=(2,2))
            self.graph.create_line(b2[0], b2[1], b3[0], b3[1], 
                                 fill="lightblue", tags="bezier_control", dash=(2,2))

    def draw_tangents(self, points, tangents):
        """Dessine les tangentes"""
        if not self.show_tangents.get() or len(points) != len(tangents):
            return
            
        self.graph.delete("tangents")
        
        scale = 50  # Échelle d'affichage
        for i, (point, tangent) in enumerate(zip(points, tangents)):
            point = np.array(point)
            if np.linalg.norm(tangent) > 0:
                end_point = point + scale * tangent / np.linalg.norm(tangent)
                self.graph.create_line(point[0], point[1], end_point[0], end_point[1],
                                     fill="red", width=2, tags="tangents", arrow=LAST)

    def compute_curvature(self, curve):
        """Calcule la courbure d'une courbe paramétrique"""
        if len(curve) < 3:
            return np.zeros(len(curve))
            
        dx = np.gradient(curve[:, 0])
        dy = np.gradient(curve[:, 1])
        d2x = np.gradient(dx)
        d2y = np.gradient(dy)
        
        curvature = np.abs(dx * d2y - dy * d2x) / (dx**2 + dy**2)**1.5
        return np.nan_to_num(curvature)

    def update_curvature_display(self):
        """Met à jour l'affichage de la courbure"""
        self.ax_curvature.clear()
        
        if self.show_curvature.get() and self.curve is not None and len(self.curve) >= 3:
            curvature = self.compute_curvature(self.curve)
            t = np.linspace(0, 1, len(self.curve))
            
            self.ax_curvature.plot(t, curvature, 'r-', linewidth=2)
            self.ax_curvature.set_title('Courbure κ(t)')
            self.ax_curvature.set_xlabel('Paramètre t')
            self.ax_curvature.set_ylabel('Courbure')
            self.ax_curvature.grid(True)
        
        self.canvas_plot.draw()

    def update_display(self, event=None):
        """Met à jour tous les éléments d'affichage"""
        points = self.get_points()
        if len(points) >= 2:
            u = self.compute_parameters(points)
            tangents = self.compute_cardinal_tangents(points, u)
            
            self.draw_curve()
            self.draw_polygon()
            self.draw_bezier_control_points(points, tangents, u)
            self.draw_tangents(points, tangents)
            self.update_curvature_display()




# ---------- Algorithmes d'interpolation ----------

def DeCasteljau(points, T):
    """Algorithme de De Casteljau pour les courbes de Bézier"""
    n = points.shape[0] - 1
    result = []
    for t in T:
        r = points.copy()
        for k in range(0, n):
            for i in range(0, n - k):
                r[i, :] = (1 - t) * r[i, :] + t * r[i + 1, :]
        result.append(r[0, :])
    return np.array(result)




def HermiteSpline(points, T):
    """Spline Hermite cubique"""
    if len(points) < 2:
        return points
    
    # Pour l'instant, on retourne une interpolation linéaire simple
    # L'implémentation complète se fait dans les méthodes de la classe
    result = []
    for t in T:
        idx = int(t * (len(points) - 1))
        if idx == len(points) - 1:
            result.append(points[-1])
        else:
            t_segment = t * (len(points) - 1) - idx
            p = (1 - t_segment) * points[idx] + t_segment * points[idx + 1]
            result.append(p)
    
    return np.array(result)




def LagrangeInterpolation(points, T):
    """Interpolation de Lagrange"""
    if len(points) < 2:
        return points
        
    n = len(points)
    result = []
    
    for t in T:
        x, y = 0, 0
        for i in range(n):
            # Polynôme de Lagrange L_i(t)
            L = 1
            for j in range(n):
                if i != j:
                    L *= (t - j/(n-1)) / (i/(n-1) - j/(n-1))
            x += points[i, 0] * L
            y += points[i, 1] * L
        result.append([x, y])
    
    return np.array(result)

if __name__ == "__main__":
    algorithms = [
        {"name": "Bézier", "algo": DeCasteljau},
        {"name": "Hermite", "algo": HermiteSpline},
        {"name": "Lagrange", "algo": LagrangeInterpolation}
    ]
    
    window = CurveEditorWindow(algorithms)
    window.mainloop()